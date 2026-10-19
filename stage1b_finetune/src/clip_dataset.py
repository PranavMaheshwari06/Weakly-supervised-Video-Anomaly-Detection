# stage1b_finetune/src/clip_dataset.py
"""
Clip-level dataset for LoRA fine-tuning.

Iterates over videos, decodes raw MP4s on-the-fly, and returns
individual clip tensors with video-level labels and teacher-score weights.

This reuses the Stage-0 video decode logic and clip index contract:
- 15 FPS timeline
- 16-frame clips
- stride 8

Each item returned is a single clip ready for the LoRA backbone.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF


@dataclass(frozen=True)
class VideoRecord:
    """Metadata for one video from videos.jsonl."""
    rel_path: str
    split: str
    group: str
    label_bin: int
    keep_every_n: int
    clip_starts: List[int]


@dataclass(frozen=True)
class ClipSample:
    """A single clip ready for training."""
    clip_tensor: torch.Tensor   # [3, 16, 224, 224] float32 in [0, 1]
    label: int                  # 0..13 class index
    teacher_weight: float       # weight for this clip's loss
    video_id: str
    clip_start: int


def load_videos_index(jsonl_path: str) -> Dict[str, VideoRecord]:
    """Load videos.jsonl into a dict keyed by rel_path."""
    records = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records[obj["rel_path"]] = VideoRecord(
                rel_path=obj["rel_path"],
                split=obj["split"],
                group=obj["group"],
                label_bin=int(obj["label_bin"]),
                keep_every_n=int(obj["keep_every_n"]),
                clip_starts=obj["clip_starts"],
            )
    return records


def load_teacher_scores_lookup(teacher_path: str) -> Dict[str, float]:
    """
    Load teacher_scores_{split}.pt and build clip_key -> score lookup.
    Clip key format: "{video_id}__s{clip_start:06d}"
    """
    obj = torch.load(teacher_path, map_location="cpu")
    clip_keys = obj["clip_keys"]
    scores = obj["scores"].float().tolist()

    lookup = {}
    for ck, sc in zip(clip_keys, scores):
        lookup[ck] = float(sc)
    return lookup


def build_canonical_label_map(
    videos_index: Dict[str, VideoRecord],
    split_video_ids: List[str],
) -> Dict[str, int]:
    """
    Build label map from training videos.
    Normal -> 0, anomaly groups sorted -> 1..13
    Must match Stage-2 label map exactly.
    """
    groups = set()
    for vid in split_video_ids:
        rec = videos_index[vid]
        if rec.label_bin == 0:
            groups.add("Normal")
        else:
            groups.add(rec.group)

    # Canonical ordering: Normal first, then sorted anomaly groups
    class_names = ["Normal"] + sorted([g for g in groups if g != "Normal"])
    label_map = {g: i for i, g in enumerate(class_names)}
    return label_map


def read_split_txt(path: str) -> List[str]:
    """Read video IDs from split file."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def make_clip_key(video_id: str, clip_start: int) -> str:
    """Must match Stage-1 format exactly."""
    return f"{video_id}__s{clip_start:06d}"


def decode_clip_frames(
    video_path: str,
    clip_start_15fps: int,
    clip_len: int,
    keep_every_n: int,
) -> Optional[torch.Tensor]:
    """
    Decode a single clip from a video file.

    Args:
        video_path: path to raw MP4
        clip_start_15fps: start index on 15 FPS timeline
        clip_len: number of frames in clip (16)
        keep_every_n: frame subsampling factor (e.g., 2 for 30->15 FPS)

    Returns:
        frames: [clip_len, H, W, 3] uint8 RGB tensor, or None on failure
    """
    # Calculate original frame indices we need
    needed_orig = [(clip_start_15fps + j) * keep_every_n for j in range(clip_len)]
    start_orig = needed_orig[0]
    end_orig = needed_orig[-1]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_orig)
    current = start_orig

    frames_dict: Dict[int, np.ndarray] = {}
    try:
        while current <= end_orig:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            if current in needed_orig:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_dict[current] = frame_rgb
            current += 1
    finally:
        cap.release()

    # Check we got all frames
    if len(frames_dict) != clip_len:
        return None

    # Stack in order
    frames_list = [frames_dict[idx] for idx in needed_orig]
    frames_np = np.stack(frames_list, axis=0)  # [T, H, W, 3]
    return torch.from_numpy(frames_np)


def transform_clip(frames_thwc: torch.Tensor, img_size: int = 224) -> torch.Tensor:
    """
    Transform raw clip frames to model input.

    Input: [16, H, W, 3] uint8
    Output: [3, 16, 224, 224] float32 in [0, 1]
    """
    # [T, H, W, 3] -> [T, 3, H, W]
    frames = frames_thwc.permute(0, 3, 1, 2).float() / 255.0

    # Resize shortest side to 224, then center crop
    t, c, h, w = frames.shape
    if h <= w:
        new_h = img_size
        new_w = int(round(w * (img_size / h)))
    else:
        new_w = img_size
        new_h = int(round(h * (img_size / w)))

    frames = torch.nn.functional.interpolate(
        frames, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    # Center crop
    _, _, h, w = frames.shape
    top = (h - img_size) // 2
    left = (w - img_size) // 2
    frames = frames[:, :, top:top + img_size, left:left + img_size]

    # [T, 3, 224, 224] -> [3, T, 224, 224]
    clip = frames.permute(1, 0, 2, 3).contiguous()
    return clip


class ClipLevelDataset:
    """
    Video-centric clip dataset for LoRA fine-tuning.

    Does NOT inherit from torch.utils.data.Dataset because we iterate
    video-by-video (not clip-by-clip) for efficient decode.

    Usage:
        dataset = ClipLevelDataset(...)
        for video_clips in dataset.iterate_epoch(epoch=0):
            # video_clips: List[ClipSample] for one video
            ...
    """

    def __init__(
        self,
        raw_root: str,
        videos_index: Dict[str, VideoRecord],
        video_ids: List[str],
        label_map: Dict[str, int],
        teacher_scores: Dict[str, float],
        clips_per_video: int = 8,
        clip_len: int = 16,
        img_size: int = 224,
        teacher_weight_temperature: float = 2.0,
        normal_clip_weight: float = 1.0,
        min_anomaly_clip_weight: float = 0.1,
        seed: int = 1337,
    ):
        self.raw_root = raw_root
        self.videos_index = videos_index
        self.video_ids = video_ids
        self.label_map = label_map
        self.teacher_scores = teacher_scores
        self.clips_per_video = clips_per_video
        self.clip_len = clip_len
        self.img_size = img_size
        self.teacher_weight_temperature = teacher_weight_temperature
        self.normal_clip_weight = normal_clip_weight
        self.min_anomaly_clip_weight = min_anomaly_clip_weight
        self.seed = seed

    def _get_label(self, video_id: str) -> int:
        rec = self.videos_index[video_id]
        if rec.label_bin == 0:
            return self.label_map["Normal"]
        return self.label_map[rec.group]

    def _sample_clip_starts(
        self, video_id: str, epoch: int
    ) -> List[int]:
        """
        Sample clip starts for a video. For anomaly videos, bias toward
        clips with high teacher scores. For normal videos, sample uniformly.
        """
        rec = self.videos_index[video_id]
        all_starts = rec.clip_starts

        if len(all_starts) <= self.clips_per_video:
            return all_starts

        # Deterministic per-video per-epoch RNG
        video_hash = abs(hash(video_id)) % (10**9)
        rng = random.Random(self.seed + epoch * 1000 + video_hash)

        if rec.label_bin == 0:
            # Normal: uniform sampling
            return sorted(rng.sample(all_starts, self.clips_per_video))

        # Anomaly: teacher-score guided sampling
        # Get scores for all clips of this video
        scored_starts = []
        for cs in all_starts:
            ck = make_clip_key(video_id, cs)
            score = self.teacher_scores.get(ck, 0.0)
            scored_starts.append((cs, score))

        # Sort by score descending, take top half as high-priority pool
        scored_starts.sort(key=lambda x: x[1], reverse=True)
        top_pool_size = max(self.clips_per_video, len(scored_starts) // 2)
        top_pool = scored_starts[:top_pool_size]

        # Sample: 50% from top pool, 50% uniform from all
        n_top = self.clips_per_video // 2
        n_uniform = self.clips_per_video - n_top

        top_chosen = rng.sample(top_pool, min(n_top, len(top_pool)))
        remaining = [s for s in scored_starts if s not in top_chosen]
        if len(remaining) >= n_uniform:
            uniform_chosen = rng.sample(remaining, n_uniform)
        else:
            uniform_chosen = remaining

        chosen = [s[0] for s in top_chosen + uniform_chosen]
        return sorted(chosen[:self.clips_per_video])

    def _compute_clip_weight(self, video_id: str, clip_start: int) -> float:
        """
        Compute loss weight for a clip based on teacher score.

        For normal videos: weight = normal_clip_weight (constant)
        For anomaly videos: weight = softmax-temperature-scaled teacher score
        """
        rec = self.videos_index[video_id]
        if rec.label_bin == 0:
            return self.normal_clip_weight

        ck = make_clip_key(video_id, clip_start)
        score = self.teacher_scores.get(ck, 0.0)

        # Normalize score: higher teacher score = more anomalous = higher weight
        # Teacher scores are "min sq dist to normal prototypes"
        # Typical range: 0.0 to ~2.0, most around 0.1-0.5
        weight = max(self.min_anomaly_clip_weight, score * self.teacher_weight_temperature)
        # Cap at 5.0 to prevent instability
        weight = min(weight, 5.0)
        return weight

    def iterate_epoch(self, epoch: int):
        """
        Yields List[ClipSample] for each video in shuffled order.
        Each list contains clips_per_video clips from one video.
        """
        # Shuffle video order per epoch
        video_hash = abs(hash(f"epoch_{epoch}")) % (10**9)
        rng = random.Random(self.seed + video_hash)
        shuffled_ids = self.video_ids[:]
        rng.shuffle(shuffled_ids)

        for video_id in shuffled_ids:
            rec = self.videos_index[video_id]
            video_path = str(Path(self.raw_root) / rec.rel_path)

            # Check video exists
            if not Path(video_path).exists():
                print(f"[WARN] Missing video: {video_path}")
                continue

            # Sample clip starts
            clip_starts = self._sample_clip_starts(video_id, epoch)
            if not clip_starts:
                continue

            # Get label for this video
            label = self._get_label(video_id)

            # Decode and transform each clip
            video_clips: List[ClipSample] = []
            for cs in clip_starts:
                frames = decode_clip_frames(
                    video_path=video_path,
                    clip_start_15fps=cs,
                    clip_len=self.clip_len,
                    keep_every_n=rec.keep_every_n,
                )
                if frames is None:
                    continue

                clip_tensor = transform_clip(frames, self.img_size)
                weight = self._compute_clip_weight(video_id, cs)

                video_clips.append(ClipSample(
                    clip_tensor=clip_tensor,
                    label=label,
                    teacher_weight=weight,
                    video_id=video_id,
                    clip_start=cs,
                ))

            if video_clips:
                yield video_clips

    def __len__(self) -> int:
        return len(self.video_ids)