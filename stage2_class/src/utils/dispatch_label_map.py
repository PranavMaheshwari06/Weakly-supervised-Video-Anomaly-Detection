"""
Emergency Dispatch Label Map for UCF-Crime.

Maps 13 anomaly classes → 3 binary dispatch labels: [police, medical, fire].
Based on real-world emergency dispatch protocols (MPDS / NFPA standards).

Multi-label: a single anomaly can trigger multiple dispatch services.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

# Each anomaly class maps to a 3-element binary vector: [police, medical, fire]
DISPATCH_MAP: Dict[str, List[int]] = {
    # Fire / Rescue primary
    "Arson":          [1, 0, 1],
    "Explosion":      [0, 1, 1],
    "RoadAccidents":  [1, 1, 1],
    # Medical primary
    "Assault":        [1, 1, 0],
    "Abuse":          [1, 1, 0],
    "Fighting":       [1, 1, 0],
    "Shooting":       [1, 1, 0],
    # Police primary
    "Burglary":       [1, 0, 0],
    "Robbery":        [1, 0, 0],
    "Shoplifting":    [1, 0, 0],
    "Stealing":       [1, 0, 0],
    "Vandalism":      [1, 0, 0],
    "Arrest":         [1, 0, 0],
}

DISPATCH_CHANNELS = ["police", "medical", "fire"]
NUM_DISPATCH = 3
PRIMARY_DISPATCH_THRESHOLD = 0.5

# Normal videos get all-zeros
NORMAL_DISPATCH = [0, 0, 0]


def get_dispatch_label(group_name: str, label_bin: int) -> List[int]:
    """Return [police, medical, fire] binary vector for a video."""
    if int(label_bin) == 0 or group_name == "Normal":
        return NORMAL_DISPATCH
    return DISPATCH_MAP.get(group_name, NORMAL_DISPATCH)


def build_dispatch_targets(
    video_ids: List[str],
    video_labels: Dict,
) -> torch.Tensor:
    """
    Build [N, 3] float tensor of dispatch labels for a list of videos.
    Used as BCE targets during training.
    """
    targets = []
    for vid in video_ids:
        info = video_labels[vid]
        group = str(info.group)
        label_bin = int(info.label_bin)
        targets.append(get_dispatch_label(group, label_bin))
    return torch.tensor(targets, dtype=torch.float32)


def dispatch_vector_to_services(
    vec: List[int],
    threshold: float = PRIMARY_DISPATCH_THRESHOLD,
) -> List[str]:
    """Convert sigmoid output or binary vector to list of service names."""
    services = []
    for i, val in enumerate(vec):
        if val >= threshold:
            services.append(DISPATCH_CHANNELS[i])
    return services
