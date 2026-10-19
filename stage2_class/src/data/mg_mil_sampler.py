from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import List

from stage2_class.src.data.mil_index import ClipRef


@dataclass(frozen=True)
class MgMilConfig:
    bag_size: int = 16
    warmup_epochs: int = 2
    mixture_uniform_ratio: float = 0.5  # remaining comes from top-scored
    top_fraction: float = 0.25          # take top 25% clips as pool


class MgMilSampler:
    def __init__(self, cfg: MgMilConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed

    def select_refs_for_video(
        self,
        video_clips: List[ClipRef],
        epoch: int,
        video_id: str,
        context: str = "train",
    ) -> List[ClipRef]:
        if len(video_clips) == 0:
            raise ValueError(f"video_clips empty for video_id={video_id}")

        rng = random.Random(stable_sampling_seed(self.seed, epoch, context, video_id))

        bag_size = self.cfg.bag_size

        # Warmup: purely uniform
        if epoch < self.cfg.warmup_epochs:
            if len(video_clips) <= bag_size:
                return list(video_clips)
            return rng.sample(video_clips, k=bag_size)

        num_uniform = int(round(bag_size * self.cfg.mixture_uniform_ratio))
        num_top = bag_size - num_uniform

        # Uniform portion
        if len(video_clips) <= num_uniform:
            uniform_refs = list(video_clips)
        else:
            uniform_refs = rng.sample(video_clips, k=num_uniform)

        already = set(r.clip_key for r in uniform_refs)

        # Top pool based on fraction
        sorted_by_score = sorted(video_clips, key=lambda r: r.teacher_score, reverse=True)
        top_pool_size = max(bag_size, int(math.ceil(len(sorted_by_score) * self.cfg.top_fraction)))
        top_pool_size = min(top_pool_size, len(sorted_by_score))
        top_pool = sorted_by_score[:top_pool_size]

        candidates = [r for r in top_pool if r.clip_key not in already]

        if len(candidates) == 0:
            remaining = [r for r in video_clips if r.clip_key not in already]
            if len(remaining) == 0:
                return uniform_refs
            if len(remaining) <= num_top:
                return uniform_refs + remaining
            return uniform_refs + rng.sample(remaining, k=num_top)

        if len(candidates) <= num_top:
            return uniform_refs + candidates

        top_refs = rng.sample(candidates, k=num_top)
        return uniform_refs + top_refs


def stable_sampling_seed(seed: int, epoch: int, context: str, video_id: str) -> int:
    """Return a process-independent seed for one video-bag selection."""
    value = f"{int(seed)}\n{int(epoch)}\n{context}\n{video_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], byteorder="big", signed=False)
