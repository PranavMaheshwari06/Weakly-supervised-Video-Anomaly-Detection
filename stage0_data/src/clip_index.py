"""
Clip index builder — produces a JSONL file mapping each video to its
valid clip start indices on the 15 FPS timeline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from stage0_data.src.video_decode import decode_video_to_tensor


@dataclass(frozen=True)
class SplitPaths:
    raw_root: Path
    split_dir: Path
    out_dir: Path


def read_split_list(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing split file: {txt_path}")

    rel_paths: List[str] = []
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        rel_paths.append(line.replace("\\", "/").lstrip("./"))

    if not rel_paths:
        raise ValueError(f"Split file is empty: {txt_path}")

    return rel_paths


def safe_float(value: float, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if not np.isfinite(v) or v <= 0:
        return default
    return v

def get_group_name(rel_path: str) -> str:
    parts = Path(rel_path).parts
    return parts[0] if parts else "UNKNOWN"


def binary_label_from_group(group: str) -> int:
    """
    For anomaly detection experiment:
    - normal folders: Training_Normal_Videos_Anomaly, Testing_Normal_Videos_Anomaly -> label 0
    - anomaly folders: Abuse, Arrest, ... -> label 1
    """
    if group in {"Training_Normal_Videos_Anomaly", "Testing_Normal_Videos_Anomaly"}:
        return 0
    return 1


def estimate_timeline_frames_metadata(
    video_path: Path,
    fps_target: float,
    fps_fallback: float = 30.0,
) -> Tuple[int, float, int, int]:
    """
    Estimate number of frames on the target timeline using OpenCV metadata.

    Returns:
        num_frames_15fps: int
        src_fps: float
        keep_every_n: int
        src_frame_count: int
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        src_fps = safe_float(cap.get(cv2.CAP_PROP_FPS), default=fps_fallback)
        src_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if src_frame_count <= 0:
        raise RuntimeError(f"OpenCV frame_count is invalid for: {video_path} (got {src_frame_count})")

    keep_every_n = max(1, int(round(src_fps / fps_target)))
    # If we keep every n-th frame, timeline length is roughly ceil(frame_count / n)
    num_frames_15fps = int((src_frame_count + keep_every_n - 1) // keep_every_n)

    return num_frames_15fps, src_fps, keep_every_n, src_frame_count


def estimate_timeline_frames_decode(
    video_path: Path,
    fps_target: float,
    max_frames: Optional[int] = None,
) -> Tuple[int, float, int, int]:
    """
    Robust but slower: decode and count frames kept by the sampling rule inside decoder.
    """
    frames, src_fps = decode_video_to_tensor(
        video_path=video_path,
        fps_target=fps_target,
        max_frames=max_frames,
        convert_rgb=True,
    )
    # Our decoder keeps every n-th frame where n = round(src_fps / fps_target)
    keep_every_n = max(1, int(round(src_fps / fps_target)))
    # We don't know exact src_frame_count without reading all frames; set to -1
    return int(frames.shape[0]), float(src_fps), int(keep_every_n), -1


def build_clip_starts(num_frames: int, clip_len: int, stride: int) -> List[int]:
    if num_frames < clip_len:
        return []
    return list(range(0, num_frames - clip_len + 1, stride))


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clip index JSONL for UCF-Crime (DMAN v2.1).")
    parser.add_argument("--fps_target", type=float, default=15.0)
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument(
        "--count_method",
        type=str,
        default="metadata",
        choices=["metadata", "decode"],
        help="How to estimate timeline frame count: metadata (fast) or decode (robust but slow).",
    )
    parser.add_argument(
        "--decode_fallback",
        action="store_true",
        help="If metadata counting fails for a video, fall back to decoding for that video.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional limit for debugging (process only first N videos total).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    paths = SplitPaths(
        raw_root=repo_root / "data" / "ucf_crime" / "raw",
        split_dir=repo_root / "data" / "ucf_crime" / "splits" / "dman_v21",
        out_dir=repo_root / "data" / "ucf_crime" / "artifacts" / "dman_v21" / "clip_index",
    )

    train_list = read_split_list(paths.split_dir / "train.txt")
    val_list = read_split_list(paths.split_dir / "val.txt")
    test_list = read_split_list(paths.split_dir / "test.txt")

    all_items: List[Tuple[str, str]] = []
    all_items += [("train", p) for p in train_list]
    all_items += [("val", p) for p in val_list]
    all_items += [("test", p) for p in test_list]

    if args.max_videos is not None:
        all_items = all_items[: args.max_videos]

    out_jsonl = paths.out_dir / "videos.jsonl"
    out_stats = paths.out_dir / "stats.json"

    records: List[dict] = []
    stats = {
        "config": {
            "fps_target": args.fps_target,
            "clip_len": args.clip_len,
            "stride": args.stride,
            "count_method": args.count_method,
            "decode_fallback": bool(args.decode_fallback),
        },
        "totals": {
            "videos": 0,
            "clips": 0,
            "by_split": {"train": 0, "val": 0, "test": 0},
            "clips_by_split": {"train": 0, "val": 0, "test": 0},
        },
        "by_group": {
            "videos_by_split": {"train": {}, "val": {}, "test": {}},
            "clips_by_split": {"train": {}, "val": {}, "test": {}},
        },
        "failures": [],
    }

    def bump_counter(d: Dict[str, int], key: str, amount: int) -> None:
        d[key] = int(d.get(key, 0)) + int(amount)

    for split_name, rel_path in all_items:
        video_path = paths.raw_root / rel_path
        group = get_group_name(rel_path)
        label = binary_label_from_group(group)

        if not video_path.exists():
            stats["failures"].append({"split": split_name, "rel_path": rel_path, "reason": "missing_file"})
            continue

        try:
            if args.count_method == "metadata":
                num_frames_15, src_fps, keep_n, src_frame_count = estimate_timeline_frames_metadata(
                    video_path=video_path,
                    fps_target=args.fps_target,
                )
            else:
                num_frames_15, src_fps, keep_n, src_frame_count = estimate_timeline_frames_decode(
                    video_path=video_path,
                    fps_target=args.fps_target,
                    max_frames=None,
                )
        except Exception as e:
            if args.decode_fallback and args.count_method == "metadata":
                try:
                    num_frames_15, src_fps, keep_n, src_frame_count = estimate_timeline_frames_decode(
                        video_path=video_path,
                        fps_target=args.fps_target,
                        max_frames=None,
                    )
                except Exception as e2:
                    stats["failures"].append(
                        {"split": split_name, "rel_path": rel_path, "reason": f"count_failed: {str(e2)}"}
                    )
                    continue
            else:
                stats["failures"].append({"split": split_name, "rel_path": rel_path, "reason": f"count_failed: {str(e)}"})
                continue

        clip_starts = build_clip_starts(num_frames=num_frames_15, clip_len=args.clip_len, stride=args.stride)

        record = {
            "rel_path": rel_path,
            "split": split_name,
            "group": group,
            "label_bin": label,
            "src_fps": float(src_fps),
            "keep_every_n": int(keep_n),
            "src_frame_count": int(src_frame_count),
            "num_frames_15fps": int(num_frames_15),
            "clip_len": int(args.clip_len),
            "stride": int(args.stride),
            "num_clips": int(len(clip_starts)),
            "clip_starts": clip_starts,
        }
        records.append(record)

        # Update stats
        stats["totals"]["videos"] += 1
        stats["totals"]["clips"] += len(clip_starts)
        stats["totals"]["by_split"][split_name] += 1
        stats["totals"]["clips_by_split"][split_name] += len(clip_starts)

        bump_counter(stats["by_group"]["videos_by_split"][split_name], group, 1)
        bump_counter(stats["by_group"]["clips_by_split"][split_name], group, len(clip_starts))

    write_jsonl(out_jsonl, records)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Clip index built")
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_stats}")
    print("Totals:", stats["totals"])
    if stats["failures"]:
        print(f"⚠ failures: {len(stats['failures'])} (see stats.json for details)")


if __name__ == "__main__":
    main()
