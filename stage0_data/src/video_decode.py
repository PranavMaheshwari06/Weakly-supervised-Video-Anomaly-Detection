"""
Video decoding utilities for the UCF-Crime pipeline.
Outputs torch.uint8 tensors in (T, H, W, 3) RGB layout at a target FPS.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class DecodeResult:
    frames: torch.Tensor   # uint8 (T, H, W, 3) RGB
    src_fps: float
    used_fps: float
    kept_every_n: int
    num_read: int
    num_kept: int


def _safe_fps(value: float, default: float = 30.0) -> float:
    """Return a finite, positive FPS value or fall back to *default*."""
    try:
        v = float(value)
    except Exception:
        return default
    return v if (np.isfinite(v) and v > 0) else default


def _decode_video(
    video_path: Path,
    fps_target: float,
    max_frames: Optional[int],
    convert_rgb: bool,
) -> Tuple[torch.Tensor, float, int, int, int]:
    """
    Core decode loop shared by the public helpers.

    Returns (frames_tensor, src_fps, keep_every_n, total_read, total_kept).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
    keep_every_n = max(1, int(round(src_fps / fps_target)))

    kept: list[np.ndarray] = []
    total_read = 0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            total_read += 1

            if (total_read - 1) % keep_every_n != 0:
                continue

            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if convert_rgb else frame_bgr
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            kept.append(frame)

            if max_frames is not None and len(kept) >= max_frames:
                break
    finally:
        cap.release()

    if not kept:
        raise RuntimeError(
            f"No frames decoded from {video_path} "
            f"(src_fps={src_fps:.1f}, keep_every_n={keep_every_n})"
        )

    frames_tensor = torch.from_numpy(np.stack(kept, axis=0))
    return frames_tensor, src_fps, keep_every_n, total_read, len(kept)


def decode_video_to_tensor(
    video_path: Union[str, Path],
    fps_target: float = 15.0,
    max_frames: Optional[int] = None,
    convert_rgb: bool = True,
) -> Tuple[torch.Tensor, float]:
    """Decode *video_path* and return ``(frames_uint8, src_fps)``."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if fps_target <= 0:
        raise ValueError(f"fps_target must be > 0, got {fps_target}")

    frames, src_fps, *_ = _decode_video(video_path, fps_target, max_frames, convert_rgb)
    return frames, src_fps


def decode_video_verbose(
    video_path: Union[str, Path],
    fps_target: float = 15.0,
    max_frames: Optional[int] = None,
    convert_rgb: bool = True,
) -> DecodeResult:
    """Like ``decode_video_to_tensor`` but returns richer metadata for sanity checks."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if fps_target <= 0:
        raise ValueError(f"fps_target must be > 0, got {fps_target}")

    frames, src_fps, keep_every_n, total_read, total_kept = _decode_video(
        video_path, fps_target, max_frames, convert_rgb,
    )
    return DecodeResult(
        frames=frames,
        src_fps=src_fps,
        used_fps=src_fps / keep_every_n,
        kept_every_n=keep_every_n,
        num_read=total_read,
        num_kept=total_kept,
    )

