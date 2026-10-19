from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml

from stage2_class.src.io.feature_store import iter_global_shards


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize each row vector to unit L2 norm.
    Computes in float32 for stability.

    x: [N, D] float16/float32
    returns: [N, D] float32
    """
    x32 = x.float()
    norms = torch.linalg.norm(x32, ord=2, dim=1, keepdim=True).clamp_min(eps)
    return x32 / norms


def min_sq_euclidean_distance_to_prototypes(
    x_unit: torch.Tensor,
    prototypes_unit: torch.Tensor,
    block_size: int = 4096,
) -> torch.Tensor:
    """
    Compute score(x) = min_k ||x - p_k||^2 for each row x.

    x_unit: [N, D] float32, unit-norm
    prototypes_unit: [K, D] float32, unit-norm

    Returns:
      scores: [N] float32
    """
    if x_unit.ndim != 2 or prototypes_unit.ndim != 2:
        raise ValueError("x_unit and prototypes_unit must be 2D tensors")

    num_points = x_unit.shape[0]
    num_prototypes = prototypes_unit.shape[0]
    device = x_unit.device

    scores = torch.empty(num_points, dtype=torch.float32, device=device)

    # Precompute prototype squared norms (should be ~1, but do it anyway safely)
    proto_sq = (prototypes_unit * prototypes_unit).sum(dim=1).view(1, num_prototypes)  # [1,K]

    start = 0
    while start < num_points:
        end = min(start + block_size, num_points)
        block = x_unit[start:end]  # [B,D]

        block_sq = (block * block).sum(dim=1, keepdim=True)  # [B,1]
        # dist^2 = ||x||^2 + ||p||^2 - 2 x·p
        dist2 = block_sq + proto_sq - 2.0 * (block @ prototypes_unit.t())  # [B,K]
        scores[start:end] = torch.min(dist2, dim=1).values
        start = end

    return scores


def compute_stats(scores: torch.Tensor) -> Dict:
    """
    scores: [N] float32 on CPU
    """
    s = scores.float().cpu()
    percentiles = [1, 5, 10, 50, 90, 95, 99]
    pct_vals = torch.quantile(s, torch.tensor([p / 100.0 for p in percentiles]))

    stats = {
        "count": int(s.numel()),
        "finite": bool(torch.isfinite(s).all().item()),
        "mean": float(s.mean().item()),
        "std": float(s.std(unbiased=False).item()),
        "min": float(s.min().item()),
        "max": float(s.max().item()),
        "percentiles": {str(p): float(v.item()) for p, v in zip(percentiles, pct_vals)},
    }
    return stats


def load_prototypes(prototypes_path: str) -> torch.Tensor:
    obj = torch.load(prototypes_path, map_location="cpu")
    if not isinstance(obj, dict) or "prototypes" not in obj:
        raise ValueError(f"Bad prototypes file (missing 'prototypes'): {prototypes_path}")

    prototypes = obj["prototypes"]
    if not torch.is_tensor(prototypes):
        raise ValueError(f"'prototypes' must be a tensor in: {prototypes_path}")

    # normalize to unit length in float32
    prototypes_unit = l2_normalize_rows(prototypes)
    if not torch.isfinite(prototypes_unit).all().item():
        raise ValueError("Non-finite values in prototypes after normalization")

    return prototypes_unit


def score_split(
    store_root: str,
    split: str,
    expected_dim: int,
    prototypes_unit: torch.Tensor,
    max_shards_to_scan: int,
    block_size: int,
) -> Tuple[Dict, Dict]:
    """
    Returns:
      saved_payload: dict to torch.save
      stats: dict with score stats
    """
    all_clip_keys: List[str] = []
    all_video_ids: List[str] = []
    all_clip_starts: List[int] = []
    all_scores: List[torch.Tensor] = []

    for shard_path, shard in iter_global_shards(
        store_root=store_root,
        split=split,
        expected_dim=expected_dim,
        max_shards_to_scan=max_shards_to_scan,
    ):
        x_unit = l2_normalize_rows(shard.global_emb)  # [N,D] float32
        scores = min_sq_euclidean_distance_to_prototypes(
            x_unit=x_unit,
            prototypes_unit=prototypes_unit,
            block_size=block_size,
        ).cpu()

        all_clip_keys.extend(shard.clip_keys)
        all_video_ids.extend(shard.video_ids)
        all_clip_starts.extend([int(v) for v in shard.clip_starts.tolist()])
        all_scores.append(scores)

    if len(all_clip_keys) == 0:
        raise ValueError(f"No clips found for split={split}")

    scores_cat = torch.cat(all_scores, dim=0).float()
    if scores_cat.shape[0] != len(all_clip_keys):
        raise ValueError("Internal error: score length mismatch with clip_keys")

    stats = compute_stats(scores_cat)

    payload = {
        "meta": {
            "stage": "stage2",
            "artifact": "teacher_scores",
            "split": split,
            "score_def": "min_sq_euclidean_to_normal_prototypes",
            "embedding_norm": "l2",
            "prototypes_source": "artifacts/stage2/memory/global_prototypes_k100.pt",
            "expected_dim": expected_dim,
        },
        "clip_keys": all_clip_keys,
        "video_ids": all_video_ids,
        "clip_starts": torch.tensor(all_clip_starts, dtype=torch.int32),
        "scores": scores_cat,  # float32 [N]
    }

    return payload, stats


def main() -> None:
    cfg_path = os.environ.get("STAGE2_CFG", "stage2_class/configs/base.yaml")
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    store_root = cfg["paths"]["feature_store_root"]
    expected_dim = int(cfg["stage1"]["global_dim"])
    max_scan = int(cfg["io"]["max_shards_to_scan"])
    out_results = cfg["io"]["write_dir_results"]

    out_teacher_dir = "artifacts/stage2/teacher"
    ensure_dir(out_teacher_dir)
    ensure_dir(out_results)

    prototypes_path = "artifacts/stage2/memory/global_prototypes_k100.pt"
    prototypes_unit = load_prototypes(prototypes_path)

    block_size = 4096  # safe CPU default

    all_stats: Dict[str, Dict] = {}
    for split in ["train", "val", "test"]:
        payload, stats = score_split(
            store_root=store_root,
            split=split,
            expected_dim=expected_dim,
            prototypes_unit=prototypes_unit,
            max_shards_to_scan=max_scan,
            block_size=block_size,
        )

        out_pt = Path(out_teacher_dir) / f"teacher_scores_{split}.pt"
        torch.save(payload, out_pt)
        all_stats[split] = stats

        print(f"[{split}] saved: {out_pt}")
        print(f"[{split}] stats: {json.dumps(stats, indent=2)}")

    out_json = Path(out_results) / "m23_teacher_score_stats.json"
    out_json.write_text(json.dumps(all_stats, indent=2))
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
