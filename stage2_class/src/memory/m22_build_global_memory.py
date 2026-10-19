from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import torch
import yaml

from stage2_class.src.io.feature_store import iter_global_shards


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x32 = x.float()
    norms = torch.linalg.norm(x32, ord=2, dim=1, keepdim=True).clamp_min(eps)
    return x32 / norms


def _load_normal_clip_index(path: str) -> Set[str]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "clip_keys" not in obj:
        raise ValueError(f"Bad normal clip index file: {path}")
    clip_keys = obj["clip_keys"]
    if not isinstance(clip_keys, list) or not clip_keys or not isinstance(clip_keys[0], str):
        raise ValueError(f"clip_keys must be List[str] in: {path}")
    return set(clip_keys)


def kmeans_torch(x: torch.Tensor, k: int, seed: int, iters: int = 25) -> Tuple[torch.Tensor, Dict]:
    """
    Simple CPU k-means for thesis-level usage.
    x: [N, D] float32
    returns:
      centers: [K, D] float32
      stats: dict
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.ndim}D")

    n, d = x.shape
    if n < k:
        raise ValueError(f"Need at least N>=K. Got N={n}, K={k}")

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # init: pick K random points
    init_idx = torch.randperm(n, generator=g)[:k]
    centers = x[init_idx].clone()

    last_inertia = None
    for t in range(iters):
        # distances: [N, K]
        # using (a-b)^2 = a^2 + b^2 - 2ab
        x2 = (x * x).sum(dim=1, keepdim=True)          # [N,1]
        c2 = (centers * centers).sum(dim=1).view(1, k) # [1,K]
        dist2 = x2 + c2 - 2.0 * (x @ centers.t())      # [N,K]

        assignments = torch.argmin(dist2, dim=1)       # [N]
        inertia = float(dist2[torch.arange(n), assignments].mean().item())

        # recompute centers
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, dtype=torch.int64)

        for i in range(n):
            a = int(assignments[i].item())
            new_centers[a] += x[i]
            counts[a] += 1

        # handle empty clusters by re-seeding from random points
        empty = (counts == 0).nonzero(as_tuple=False).view(-1)
        if empty.numel() > 0:
            reseed_idx = torch.randperm(n, generator=g)[: empty.numel()]
            for j, cluster_id in enumerate(empty.tolist()):
                new_centers[cluster_id] = x[reseed_idx[j]]
                counts[cluster_id] = 1

        centers = new_centers / counts.clamp_min(1).unsqueeze(1)

        if last_inertia is not None and abs(last_inertia - inertia) < 1e-6:
            last_inertia = inertia
            break
        last_inertia = inertia

    stats = {
        "k": k,
        "iters": iters,
        "final_inertia_mean_sqdist": last_inertia,
        "empty_clusters_fixed": int((counts == 0).sum().item()),
        "counts_min": int(counts.min().item()),
        "counts_max": int(counts.max().item()),
    }
    return centers, stats


def main() -> None:
    cfg_path = os.environ.get("STAGE2_CFG", "stage2_class/configs/base.yaml")
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    store_root = cfg["paths"]["feature_store_root"]
    expected_dim = int(cfg["stage1"]["global_dim"])
    max_scan = int(cfg["io"]["max_shards_to_scan"])
    seed = int(cfg.get("seed", 1337))

    normal_index_path = "artifacts/stage2/preprocess/train_normal_clip_index.pt"
    normal_clip_keys = _load_normal_clip_index(normal_index_path)

    out_results = cfg["io"]["write_dir_results"]
    out_artifacts = "artifacts/stage2/memory"
    _ensure_dir(out_results)
    _ensure_dir(out_artifacts)

    # Collect embeddings for normal clips
    collected: List[torch.Tensor] = []
    found = 0

    for shard_path, shard in iter_global_shards(
        store_root=store_root,
        split="train",
        expected_dim=expected_dim,
        max_shards_to_scan=max_scan,
    ):
        # mask by membership in normal clip set
        mask = [ck in normal_clip_keys for ck in shard.clip_keys]
        if not any(mask):
            continue

        mask_t = torch.tensor(mask, dtype=torch.bool)
        emb = shard.global_emb[mask_t]   # float16 [M,768]
        collected.append(emb)
        found += int(emb.shape[0])

    if found == 0:
        raise ValueError("Found 0 embeddings for normal clips. Index/shards mismatch.")

    x = torch.cat(collected, dim=0)  # float16 [N,768]
    if x.shape[1] != expected_dim:
        raise ValueError(f"Dim mismatch: {x.shape[1]} vs {expected_dim}")

    # normalize
    x_norm = l2_normalize(x)  # float32 [N,768]

    # k-means
    k = 100
    centers, km_stats = kmeans_torch(x_norm, k=k, seed=seed, iters=25)

    # normalize centers as well (for clean cosine-like distance later)
    centers = l2_normalize(centers)

    # save artifact
    out_pt = Path(out_artifacts) / f"global_prototypes_k{k}.pt"
    torch.save(
        {
            "meta": {
                "stage": "stage2",
                "artifact": "global_prototypes",
                "k": k,
                "dim": expected_dim,
                "seed": seed,
                "source": "train_normal_clips",
                "normal_index_path": normal_index_path,
                "normalization": "l2",
            },
            "prototypes": centers.half(),  # store fp16 to save space
            "kmeans_stats": km_stats,
            "num_points_used": int(x_norm.shape[0]),
        },
        out_pt,
    )

    # sanity summary
    proto_fp32 = centers.float()
    summary = {
        "normal_index_path": normal_index_path,
        "num_normal_clip_keys_in_index": len(normal_clip_keys),
        "num_embeddings_found_in_shards": found,
        "prototype_shape": list(proto_fp32.shape),
        "prototype_finite": bool(torch.isfinite(proto_fp32).all().item()),
        "prototype_norm_mean": float(torch.linalg.norm(proto_fp32, dim=1).mean().item()),
        "prototype_norm_min": float(torch.linalg.norm(proto_fp32, dim=1).min().item()),
        "prototype_norm_max": float(torch.linalg.norm(proto_fp32, dim=1).max().item()),
        "kmeans": km_stats,
        "saved_prototypes_path": str(out_pt),
    }

    out_json = Path(out_results) / "m22_memory_build.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_pt}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
