from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from stage2_class.src.data.mil_index import ClipRef
from stage2_class.src.io.videos_index import VideoLabel


@dataclass(frozen=True)
class MilBag:
    video_id: str
    embeddings: torch.Tensor      # [M, D] float32
    teacher_scores: torch.Tensor  # [M] float32
    label_bin: int
    group: str


class MilVideoDataset(Dataset):
    def __init__(
        self,
        video_ids: List[str],
        video_to_clips: Dict[str, List[ClipRef]],
        video_labels: Dict[str, VideoLabel],
        embedding_dim: int,
    ) -> None:
        self.video_ids = video_ids
        self.video_to_clips = video_to_clips
        self.video_labels = video_labels
        self.embedding_dim = embedding_dim

        # shard cache to avoid re-loading the same shard repeatedly
        self._cached_shard_path: Optional[str] = None
        self._cached_shard_obj: Optional[Dict] = None

        for vid in self.video_ids:
            if vid not in self.video_to_clips:
                raise ValueError(f"video_id missing clips index: {vid}")
            if vid not in self.video_labels:
                raise ValueError(f"video_id missing labels index: {vid}")

    def __len__(self) -> int:
        return len(self.video_ids)

    def _load_shard(self, shard_path: str) -> Dict:
        if self._cached_shard_path == shard_path and self._cached_shard_obj is not None:
            return self._cached_shard_obj

        obj = torch.load(shard_path, map_location="cpu")
        if not isinstance(obj, dict) or "global" not in obj:
            raise ValueError(f"Bad shard file: {shard_path}")

        self._cached_shard_path = shard_path
        self._cached_shard_obj = obj
        return obj

    def get_video_clips(self, video_id: str) -> List[ClipRef]:
        return self.video_to_clips[video_id]

    def build_bag_from_refs(self, video_id: str, refs: List[ClipRef]) -> MilBag:
        if len(refs) == 0:
            raise ValueError(f"Empty refs for video_id={video_id}")

        embeddings_list: List[torch.Tensor] = []
        scores_list: List[float] = []

        for ref in refs:
            shard = self._load_shard(ref.shard_path)
            global_emb = shard["global"]  # float16 [N,768]
            emb = global_emb[ref.local_index]  # float16 [768]
            embeddings_list.append(emb)
            scores_list.append(ref.teacher_score)

        emb_stack = torch.stack(embeddings_list, dim=0).float()  # [M,768] float32
        score_tensor = torch.tensor(scores_list, dtype=torch.float32)

        lab = self.video_labels[video_id]
        return MilBag(
            video_id=video_id,
            embeddings=emb_stack,
            teacher_scores=score_tensor,
            label_bin=int(lab.label_bin),
            group=str(lab.group),
        )

    def __getitem__(self, idx: int) -> Tuple[str, int, str]:
        """
        We do not return embeddings here because sampling is dynamic (epoch-dependent).
        Training loop will:
          - get video_id from dataset
          - sampler chooses refs
          - dataset.build_bag_from_refs() loads embeddings
        """
        vid = self.video_ids[idx]
        lab = self.video_labels[vid]
        return vid, int(lab.label_bin), str(lab.group)
