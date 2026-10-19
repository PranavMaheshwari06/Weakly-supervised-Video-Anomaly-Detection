from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import yaml

from stage2_class.src.io.feature_store import iter_global_shards
from stage2_class.src.io.split_reader import read_split_txt
from stage2_class.src.io.videos_index import load_videos_index


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x: [N, D] float16/float32 on CPU
    returns: [N, D] float32 (we keep fp32 for numerical stability)
    """
    x32 = x.float()
    norms = torch.linalg.norm(x32, ord=2, dim=1, keepdim=True).clamp_min(eps)
    return x32 / norms


def main() -> None:
    cfg_path = os.environ.get("STAGE2_CFG", "stage2_class/configs/base.yaml")
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    store_root = cfg["paths"]["feature_store_root"]
    splits_root = cfg["paths"]["splits_root"]
    videos_index_path = cfg["paths"]["videos_index_jsonl"]
    expected_dim = int(cfg["stage1"]["global_dim"])

    out_results = cfg["io"]["write_dir_results"]
    out_logs = cfg["io"]["write_dir_logs"]
    out_artifacts = "artifacts/stage2/preprocess"

    _ensure_dir(out_results)
    _ensure_dir(out_logs)
    _ensure_dir(out_artifacts)

    videos_index = load_videos_index(videos_index_path)

    train_split_path = str(Path(splits_root) / "train.txt")
    train_video_ids = set(read_split_txt(train_split_path))

    # normal train videos (video-level label_bin == 0)
    normal_train_videos: Set[str] = set()
    for vid in train_video_ids:
        if vid not in videos_index:
            raise ValueError(f"Train video missing in videos_index: {vid}")
        if videos_index[vid].label_bin == 0:
            normal_train_videos.add(vid)

    if len(normal_train_videos) == 0:
        raise ValueError("No normal train videos found. Something is wrong with labels/splits.")

    normal_clip_keys: List[str] = []
    normal_clip_starts: List[int] = []
    normal_video_ids: List[str] = []

    total_train_clips = 0
    total_train_normal_clips = 0

    # iterate through TRAIN shards only and select clips belonging to normal videos
    for shard_path, shard in iter_global_shards(
        store_root=store_root,
        split="train",
        expected_dim=expected_dim,
        max_shards_to_scan=int(cfg["io"]["max_shards_to_scan"]),
    ):
        total_train_clips += len(shard.clip_keys)

        for ck, vid, cs in zip(shard.clip_keys, shard.video_ids, shard.clip_starts.tolist()):
            if vid in normal_train_videos:
                normal_clip_keys.append(ck)
                normal_video_ids.append(vid)
                normal_clip_starts.append(int(cs))
                total_train_normal_clips += 1

    if total_train_normal_clips == 0:
        raise ValueError("No normal clips selected from train shards. Check video_id matching.")

    # Save selection (no embeddings yet, just identifiers)
    out_pt = Path(out_artifacts) / "train_normal_clip_index.pt"
    torch.save(
        {
            "meta": {
                "split": "train",
                "selection": "video_label_bin==0",
                "expected_dim": expected_dim,
            },
            "clip_keys": normal_clip_keys,
            "video_ids": normal_video_ids,
            "clip_starts": torch.tensor(normal_clip_starts, dtype=torch.int32),
        },
        out_pt,
    )

    # Write summary stats for thesis logging
    summary = {
        "train_split_videos": len(train_video_ids),
        "normal_train_videos": len(normal_train_videos),
        "total_train_clips_seen": total_train_clips,
        "total_train_normal_clips_selected": total_train_normal_clips,
        "saved_index_path": str(out_pt),
    }

    out_json = Path(out_results) / "m21_normal_selection.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_pt}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
