"""
Minimal clip dataset and spatial transforms for the UCF-Crime data pipeline.
Returns clips as (3, T, 224, 224) float32 tensors in [0, 1].
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from stage0_data.src.video_decode import decode_video_to_tensor


@dataclass(frozen=True)
class ClipItem:
    rel_path: str
    split: str
    group: str
    label_bin: int
    start_15fps: int
    clip_len: int


def repo_root_from_file() -> Path:
    # .../stage0_data/src/dataloader.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def read_split_list(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing split file: {txt_path}")
    lines = []
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(line.replace("\\", "/").lstrip("./"))
    if not lines:
        raise ValueError(f"Split file is empty: {txt_path}")
    return lines


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing jsonl file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _resize_shorter_side(frames_tchw: torch.Tensor, short_side: int) -> torch.Tensor:
    """
    frames_tchw: (T, C, H, W) float32
    Resize so that min(H, W) == short_side, preserving aspect ratio.
    """
    assert frames_tchw.ndim == 4, f"Expected 4D (T,C,H,W), got {tuple(frames_tchw.shape)}"
    t, c, h, w = frames_tchw.shape

    if h <= w:
        new_h = short_side
        new_w = int(round(w * (short_side / h)))
    else:
        new_w = short_side
        new_h = int(round(h * (short_side / w)))

    # treat batch (T, C, H, W) — interpolate keeps aspect ratio
    return F.interpolate(frames_tchw, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _center_crop(frames_tchw: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    frames_tchw: (T, C, H, W)
    """
    t, c, h, w = frames_tchw.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot center-crop {crop_size} from {(h, w)}")

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return frames_tchw[:, :, top : top + crop_size, left : left + crop_size]


def basic_clip_transform(frames_thwc_uint8: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    """
    Convert decoded frames into model-ready clip tensor.

    Input:
      frames_thwc_uint8: (T, H, W, 3) uint8 RGB

    Output:
      clip_cthw: (3, T, out_size, out_size) float32 in [0, 1]
    """
    assert frames_thwc_uint8.dtype == torch.uint8, f"Expected uint8, got {frames_thwc_uint8.dtype}"
    assert frames_thwc_uint8.ndim == 4 and frames_thwc_uint8.shape[-1] == 3

    frames_tchw = frames_thwc_uint8.permute(0, 3, 1, 2).contiguous().float() / 255.0

    frames_tchw = _resize_shorter_side(frames_tchw, short_side=256)
    frames_tchw = _center_crop(frames_tchw, crop_size=out_size)

    return frames_tchw.permute(1, 0, 2, 3).contiguous()  # (3, T, H, W)


def load_clip_items_from_index(
    index_jsonl: Path,
    allowed_splits: Optional[set[str]] = None,
    max_videos: Optional[int] = None,
    rng_seed: int = 123,
) -> List[ClipItem]:
    """
    Builds a small list of ClipItem objects from videos.jsonl.
    For Stage 0 sanity we do NOT expand all clips. We sample one clip per video.

    Strategy:
    - iterate videos.jsonl
    - for each record: randomly pick one start from clip_starts (deterministic seed)
    - store that ClipItem

    
    """
    rng = random.Random(rng_seed)
    items: List[ClipItem] = []
    seen_videos = 0

    for rec in iter_jsonl(index_jsonl):
        split = rec["split"]
        if allowed_splits is not None and split not in allowed_splits:
            continue

        starts = rec.get("clip_starts", [])
        if not starts:
            # skip videos too short for a clip
            continue

        chosen_start = rng.choice(starts)

        items.append(
            ClipItem(
                rel_path=rec["rel_path"],
                split=split,
                group=rec.get("group", "UNKNOWN"),
                label_bin=int(rec.get("label_bin", -1)),
                start_15fps=int(chosen_start),
                clip_len=int(rec.get("clip_len", 16)),
            )
        )
        seen_videos += 1
        if max_videos is not None and seen_videos >= max_videos:
            break

    if not items:
        raise RuntimeError("No ClipItems created. Check index_jsonl and allowed_splits.")
    return items


class SimpleVideoClipDataset(torch.utils.data.Dataset):
    """
    Minimal dataset that returns exactly one clip per item.
    """

    def __init__(
        self,
        raw_root: Path,
        clip_items: List[ClipItem],
        fps_target: float = 15.0,
        out_size: int = 224,
    ) -> None:
        self.raw_root = raw_root
        self.clip_items = clip_items
        self.fps_target = float(fps_target)
        self.out_size = int(out_size)

    def __len__(self) -> int:
        return len(self.clip_items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.clip_items[idx]
        video_path = self.raw_root / item.rel_path

        frames_thwc, src_fps = decode_video_to_tensor(
            video_path=video_path,
            fps_target=self.fps_target,
            max_frames=None,
            convert_rgb=True,
        )

        start = item.start_15fps
        end = start + item.clip_len

        if end > frames_thwc.shape[0]:
            raise RuntimeError(
                f"Clip slice out of range for {item.rel_path}. "
                f"start={start}, end={end}, total_frames={frames_thwc.shape[0]}"
            )

        clip_frames = frames_thwc[start:end]  # (T, H, W, 3) uint8
        clip_cthw = basic_clip_transform(clip_frames, out_size=self.out_size)

        return {
            "clip": clip_cthw,                 # (3, T, 224, 224) float32
            "label_bin": item.label_bin,       # 0 normal, 1 anomaly
            "split": item.split,
            "rel_path": item.rel_path,
            "start_15fps": item.start_15fps,
            "src_fps": float(src_fps),
        }
