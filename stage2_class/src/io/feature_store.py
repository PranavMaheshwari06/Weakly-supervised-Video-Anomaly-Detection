from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class GlobalShard:
    meta: Dict
    clip_keys: List[str]
    video_ids: List[str]
    clip_starts: torch.Tensor  # int32 [N]
    labels: torch.Tensor       # int64 [N]
    global_emb: torch.Tensor   # float16 [N, 768]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_pt_dict(path: str) -> Dict:
    obj = torch.load(path, map_location="cpu")
    _require(isinstance(obj, dict), f"{path}: expected dict, got {type(obj)}")
    return obj


def _validate_global_shard(obj: Dict, expected_dim: int, path: str) -> GlobalShard:
    required_keys = ["meta", "clip_keys", "video_ids", "clip_starts", "labels", "global"]
    for k in required_keys:
        _require(k in obj, f"{path}: missing key '{k}'")

    clip_keys = obj["clip_keys"]
    video_ids = obj["video_ids"]
    clip_starts = obj["clip_starts"]
    labels = obj["labels"]
    global_emb = obj["global"]

    _require(isinstance(clip_keys, list) and isinstance(clip_keys[0], str), f"{path}: clip_keys must be List[str]")
    _require(isinstance(video_ids, list) and isinstance(video_ids[0], str), f"{path}: video_ids must be List[str]")

    _require(torch.is_tensor(clip_starts), f"{path}: clip_starts must be tensor")
    _require(torch.is_tensor(labels), f"{path}: labels must be tensor")
    _require(torch.is_tensor(global_emb), f"{path}: global must be tensor")

    _require(clip_starts.dtype == torch.int32, f"{path}: clip_starts dtype must be int32, got {clip_starts.dtype}")
    _require(labels.dtype == torch.int64, f"{path}: labels dtype must be int64, got {labels.dtype}")
    _require(global_emb.dtype == torch.float16, f"{path}: global dtype must be float16, got {global_emb.dtype}")

    n = len(clip_keys)
    _require(len(video_ids) == n, f"{path}: video_ids length mismatch: {len(video_ids)} vs {n}")
    _require(clip_starts.shape == (n,), f"{path}: clip_starts shape mismatch: {tuple(clip_starts.shape)} vs ({n},)")
    _require(labels.shape == (n,), f"{path}: labels shape mismatch: {tuple(labels.shape)} vs ({n},)")
    _require(global_emb.ndim == 2, f"{path}: global must be 2D, got {global_emb.ndim}D")
    _require(global_emb.shape[0] == n, f"{path}: global N mismatch: {global_emb.shape[0]} vs {n}")
    _require(global_emb.shape[1] == expected_dim, f"{path}: global dim mismatch: {global_emb.shape[1]} vs {expected_dim}")

    # light sanity: finite check on a small slice
    sample = global_emb[: min(2048, n)]
    _require(torch.isfinite(sample).all().item(), f"{path}: global has non-finite values")

    return GlobalShard(
        meta=obj["meta"],
        clip_keys=clip_keys,
        video_ids=video_ids,
        clip_starts=clip_starts,
        labels=labels,
        global_emb=global_emb,
    )


def iter_global_shards(
    store_root: str,
    split: str,
    expected_dim: int,
    max_shards_to_scan: int = -1,
) -> Iterator[Tuple[str, GlobalShard]]:
    pattern = str(Path(store_root) / f"global_{split}_shard_*.pt")
    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found for split={split} with pattern: {pattern}")

    if max_shards_to_scan is not None and max_shards_to_scan > 0:
        shard_paths = shard_paths[:max_shards_to_scan]

    for shard_path in shard_paths:
        obj = _load_pt_dict(shard_path)
        shard = _validate_global_shard(obj, expected_dim=expected_dim, path=shard_path)
        yield shard_path, shard
