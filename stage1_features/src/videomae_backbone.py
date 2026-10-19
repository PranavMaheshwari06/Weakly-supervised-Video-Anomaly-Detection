"""
Frozen VideoMAE backbone wrapper for feature extraction.
Outputs global [B, 768] and patch [B, 196, 768] features.
"""

from __future__ import annotations
from typing import Tuple

import torch
from transformers import VideoMAEModel, VideoMAEImageProcessor

# Backbone constants (MCG-NJU/videomae-base)
BACKBONE_NAME = "MCG-NJU/videomae-base"
CLIP_LEN = 16
IMG_SIZE = 224
HIDDEN_DIM = 768
NUM_TOKENS_NO_CLS = 1568   # 8 temporal groups * 196 spatial patches
TEMPORAL_GROUPS = 8
SPATIAL_PATCHES = 196


def load_videomae(device: str) -> Tuple[VideoMAEModel, VideoMAEImageProcessor, None]:
    """Load pretrained VideoMAE in frozen eval mode. Returns (model, processor, None)."""
    model = VideoMAEModel.from_pretrained(BACKBONE_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    processor = VideoMAEImageProcessor.from_pretrained(BACKBONE_NAME)
    model.to(device)
    # third return kept as None for backwards compat with callers expecting a 3-tuple
    return model, processor, None


def _normalize_bcthw(clips: torch.Tensor, processor: VideoMAEImageProcessor) -> torch.Tensor:
    """Normalize (B,C,T,H,W) float32 clips using processor mean/std."""
    mean = torch.tensor(processor.image_mean, dtype=torch.float32, device=clips.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(processor.image_std, dtype=torch.float32, device=clips.device).view(1, 3, 1, 1, 1)
    return (clips - mean) / std


def extract_batch(
    model: VideoMAEModel,
    processor: VideoMAEImageProcessor,
    clips_bcthw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from a batch of clips.

    Args:
        clips_bcthw: (B, 3, 16, 224, 224) float32 in [0,1]

    Returns:
        global_f32: (B, 768) — mean-pooled token features
        patch_f32:  (B, 196, 768) — time-averaged spatial patch features
    """
    device = next(model.parameters()).device
    clips_bcthw = clips_bcthw.to(device, non_blocking=True)

    assert clips_bcthw.shape[1:] == (3, CLIP_LEN, IMG_SIZE, IMG_SIZE), \
        f"Bad clip shape: {tuple(clips_bcthw.shape)}"

    pixel_values = _normalize_bcthw(clips_bcthw, processor)
    # HF VideoMAE expects (B, T, C, H, W)
    pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()

    with torch.no_grad():
        tokens = model(pixel_values=pixel_values).last_hidden_state

    B, N, D = tokens.shape

    # drop CLS token if present (1569 -> 1568)
    if N == NUM_TOKENS_NO_CLS + 1:
        tokens = tokens[:, 1:, :]
        N = NUM_TOKENS_NO_CLS

    assert N == NUM_TOKENS_NO_CLS and D == HIDDEN_DIM, \
        f"Unexpected tokens shape: ({N}, {D}), expected ({NUM_TOKENS_NO_CLS}, {HIDDEN_DIM})"

    global_f32 = tokens.mean(dim=1)                                       # (B, 768)
    patch_f32 = tokens.view(B, TEMPORAL_GROUPS, SPATIAL_PATCHES, HIDDEN_DIM).mean(dim=1)  # (B, 196, 768)

    return global_f32, patch_f32