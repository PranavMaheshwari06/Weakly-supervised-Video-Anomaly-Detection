from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from stage2_class.src.io.feature_store import iter_global_shards


@dataclass(frozen=True)
class ClipRef:
    clip_key: str
    clip_start: int
    teacher_score: float
    shard_path: str
    local_index: int


def load_teacher_scores(teacher_scores_path: str) -> Dict[str, float]:
    obj = torch.load(teacher_scores_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"teacher scores file must be dict: {teacher_scores_path}")
    if "clip_keys" not in obj or "scores" not in obj:
        raise ValueError(f"teacher scores missing keys in: {teacher_scores_path}")

    clip_keys = obj["clip_keys"]
    scores = obj["scores"]

    if not isinstance(clip_keys, list) or not torch.is_tensor(scores):
        raise ValueError("teacher scores format invalid: clip_keys must be list, scores must be tensor")

    if len(clip_keys) != int(scores.shape[0]):
        raise ValueError("teacher scores length mismatch")

    out: Dict[str, float] = {}
    for ck, sc in zip(clip_keys, scores.tolist()):
        out[ck] = float(sc)
    return out


def build_clip_lookup_for_split(
    store_root: str,
    split: str,
    expected_dim: int,
    max_shards_to_scan: int = -1,
) -> Dict[str, Tuple[str, int, str, int]]:
    """
    Returns mapping:
      clip_key -> (video_id, clip_start, shard_path, local_index)
    """
    lookup: Dict[str, Tuple[str, int, str, int]] = {}

    for shard_path, shard in iter_global_shards(
        store_root=store_root,
        split=split,
        expected_dim=expected_dim,
        max_shards_to_scan=max_shards_to_scan,
    ):
        for local_idx, (ck, vid) in enumerate(zip(shard.clip_keys, shard.video_ids)):
            if ck in lookup:
                raise ValueError(f"Duplicate clip_key in split={split}: {ck}")

            cs = int(shard.clip_starts[local_idx].item())
            lookup[ck] = (vid, cs, shard_path, local_idx)

    return lookup


def build_video_to_clips_index(
    clip_lookup: Dict[str, Tuple[str, int, str, int]],
    teacher_scores: Dict[str, float],
) -> Dict[str, List[ClipRef]]:
    """
    video_id -> list of ClipRef (one per clip)
    """
    video_to_clips: Dict[str, List[ClipRef]] = {}

    missing_scores = 0
    for ck, (vid, cs, shard_path, local_idx) in clip_lookup.items():
        if ck not in teacher_scores:
            missing_scores += 1
            continue

        ref = ClipRef(
            clip_key=ck,
            clip_start=cs,
            teacher_score=float(teacher_scores[ck]),
            shard_path=shard_path,
            local_index=int(local_idx),
        )
        video_to_clips.setdefault(vid, []).append(ref)

    if missing_scores != 0:
        raise ValueError(f"Missing teacher scores for {missing_scores} clips (should be 0)")

    # sort per video by time (useful for debugging / reproducibility)
    for vid in video_to_clips:
        video_to_clips[vid].sort(key=lambda r: r.clip_start)

    return video_to_clips
