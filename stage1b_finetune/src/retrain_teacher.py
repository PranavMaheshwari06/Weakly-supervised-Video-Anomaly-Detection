"""
Re-compute teacher scores using LoRA-adapted features.

After extracting new features with the LoRA backbone, we need fresh
teacher scores (distance to normal prototypes) before Stage-2 MIL training.

This script:
1. Builds new normal prototypes (k=100 k-means) from LoRA features
2. Scores all clips (train/val/test) against new prototypes
3. Outputs teacher_scores_{split}.pt in same format as original

Usage:
    python -m stage1b_finetune.src.retrain_teacher \
        --cfg stage1b_finetune/configs/lora_finetune.yaml
"""

from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import yaml


def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x32 = x.float()
    norms = torch.linalg.norm(x32, ord=2, dim=1, keepdim=True).clamp_min(eps)
    return x32 / norms


def kmeans_torch(x: torch.Tensor, k: int, seed: int, iters: int = 25) -> torch.Tensor:
    """Simple CPU k-means, returns [K, D] centers."""
    import random
    n, d = x.shape
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    init_idx = torch.randperm(n, generator=g)[:k]
    centers = x[init_idx].clone()

    for _ in range(iters):
        # Assign
        x2 = (x * x).sum(dim=1, keepdim=True)
        c2 = (centers * centers).sum(dim=1).view(1, k)
        dist2 = x2 + c2 - 2.0 * (x @ centers.t())
        assignments = torch.argmin(dist2, dim=1)

        # Update
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, dtype=torch.int64)
        for i in range(n):
            a = int(assignments[i].item())
            new_centers[a] += x[i]
            counts[a] += 1

        # Handle empty clusters
        empty = (counts == 0).nonzero(as_tuple=False).view(-1)
        if empty.numel() > 0:
            reseed_idx = torch.randperm(n, generator=g)[:empty.numel()]
            for j, cluster_id in enumerate(empty.tolist()):
                new_centers[cluster_id] = x[reseed_idx[j]]
                counts[cluster_id] = 1

        centers = new_centers / counts.clamp_min(1).unsqueeze(1)

    return l2_normalize_rows(centers)


def iter_global_shards(feature_dir: str, split: str):
    """Load all shard files for a split."""
    pattern = os.path.join(feature_dir, f"global_{split}_shard_*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shards found: {pattern}")

    for p in paths:
        obj = torch.load(p, map_location="cpu")
        yield obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="stage1b_finetune/configs/lora_finetune.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    seed = int(cfg["seed"])
    feature_dir = cfg["paths"]["lora_feature_store"]

    out_teacher_dir = "artifacts/stage2/teacher_lora"
    out_memory_dir = "artifacts/stage2/memory_lora"
    os.makedirs(out_teacher_dir, exist_ok=True)
    os.makedirs(out_memory_dir, exist_ok=True)

    # Step 1: Collect normal train embeddings
    print("Collecting normal train embeddings...")
    normal_embeddings = []
    for shard in iter_global_shards(feature_dir, "train"):
        labels = shard["labels"]
        mask = labels == 0  # Normal videos have label_bin == 0
        if mask.any():
            normal_embeddings.append(shard["global"][mask])

    x_normal = torch.cat(normal_embeddings, dim=0).float()
    x_normal = l2_normalize_rows(x_normal)
    print(f"Normal embeddings: {x_normal.shape}")

    # Step 2: K-means prototypes
    print("Computing prototypes (k=100)...")
    prototypes = kmeans_torch(x_normal, k=100, seed=seed)

    proto_path = os.path.join(out_memory_dir, "global_prototypes_k100.pt")
    torch.save({
        "meta": {"k": 100, "dim": 768, "source": "lora_normal_train", "seed": seed},
        "prototypes": prototypes.half(),
    }, proto_path)
    print(f"Saved: {proto_path}")

    # Step 3: Score all splits
    for split in ["train", "val", "test"]:
        print(f"Scoring {split}...")
        all_clip_keys = []
        all_video_ids = []
        all_clip_starts = []
        all_scores = []

        for shard in iter_global_shards(feature_dir, split):
            x = l2_normalize_rows(shard["global"].float())

            # min squared euclidean distance to prototypes
            x2 = (x * x).sum(dim=1, keepdim=True)
            p2 = (prototypes * prototypes).sum(dim=1).view(1, -1)
            dist2 = x2 + p2 - 2.0 * (x @ prototypes.t())
            scores = torch.min(dist2, dim=1).values

            all_clip_keys.extend(shard["clip_keys"])
            all_video_ids.extend(shard["video_ids"])
            all_clip_starts.extend(shard["clip_starts"].tolist())
            all_scores.append(scores)

        scores_cat = torch.cat(all_scores, dim=0)

        payload = {
            "meta": {
                "stage": "stage2",
                "artifact": "teacher_scores",
                "split": split,
                "score_def": "min_sq_euclidean_to_lora_normal_prototypes",
                "embedding_norm": "l2",
                "prototypes_source": proto_path,
                "expected_dim": 768,
            },
            "clip_keys": all_clip_keys,
            "video_ids": all_video_ids,
            "clip_starts": torch.tensor(all_clip_starts, dtype=torch.int32),
            "scores": scores_cat,
        }

        out_path = os.path.join(out_teacher_dir, f"teacher_scores_{split}.pt")
        torch.save(payload, out_path)
        print(f"  Saved: {out_path} | clips={len(all_clip_keys)} | score_mean={scores_cat.mean():.4f}")

    print("\nDone! Teacher scores recomputed for LoRA features.")
    print(f"Update your Stage-2 config to use teacher_dir: {out_teacher_dir}")


if __name__ == "__main__":
    main()