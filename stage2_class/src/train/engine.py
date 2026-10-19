"""
Stage-2 MIL Shared Training Engine & Utilities.
Consolidates boilerplate data loading, bag collation, and batch building across MIL models.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import torch

from stage2_class.src.data.mg_mil_sampler import MgMilSampler
from stage2_class.src.data.mil_dataset import MilVideoDataset
from stage2_class.src.utils.dispatch_label_map import get_dispatch_label


def canonical_group_name(group: str, label_bin: int) -> str:
    """
    UCF-Crime uses different normal group names across splits.
    Forces one canonical 'Normal' group name.
    """
    if int(label_bin) == 0:
        return "Normal"
    return str(group)


def collate_video_ids(batch: List[Tuple[str, int, str]]) -> Dict[str, object]:
    """Collate function for DataLoader returning video IDs, groups, and binary labels."""
    return {
        "video_ids": [item[0] for item in batch],
        "groups": [item[2] for item in batch],
        "label_bin": [int(item[1]) for item in batch],
    }


def build_dispatch_bags_for_batch(
    dataset: MilVideoDataset,
    sampler: MgMilSampler,
    epoch: int,
    video_ids: List[str],
    device: torch.device,
    inject_teacher_score: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds MIL feature bags for dispatch multi-label classification.
    
    Returns:
      x_batch:           [B, M, D]
      mask_batch:        [B, M]
      y_binary_batch:    [B]     0=Normal, 1=Anomaly
      y_dispatch_batch:  [B, 3]  multi-label float targets
    """
    bag_embeddings = []
    bag_masks = []
    bag_binary = []
    bag_dispatch = []

    bag_size = sampler.cfg.bag_size

    for video_id in video_ids:
        clips = dataset.get_video_clips(video_id)
        refs = sampler.select_refs_for_video(clips, epoch=epoch, video_id=video_id)

        if len(refs) < bag_size:
            while len(refs) < bag_size:
                refs.append(refs[-1])
        refs = refs[:bag_size]

        bag = dataset.build_bag_from_refs(video_id, refs)
        x = bag.embeddings

        if inject_teacher_score:
            ts = bag.teacher_scores.to(dtype=x.dtype).unsqueeze(1)
            x = torch.cat([x, ts], dim=1)

        info = dataset.video_labels[video_id]
        binary = 0 if int(info.label_bin) == 0 else 1
        dispatch = get_dispatch_label(str(info.group), int(info.label_bin))

        bag_embeddings.append(x)
        bag_masks.append(torch.ones(x.shape[0], dtype=torch.bool))
        bag_binary.append(binary)
        bag_dispatch.append(dispatch)

    x_batch = torch.stack(bag_embeddings).to(device)
    mask_batch = torch.stack(bag_masks).to(device)
    y_binary = torch.tensor(bag_binary, dtype=torch.int64, device=device)
    y_dispatch = torch.tensor(bag_dispatch, dtype=torch.float32, device=device)

    return x_batch, mask_batch, y_binary, y_dispatch
