from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from stage2_class.src.io.videos_index import VideoLabel


def build_label_map_from_train(
    train_video_ids: List[str],
    video_labels: Dict[str, VideoLabel],
) -> Dict[str, int]:
    """
    Build mapping group -> class_id.
    We enforce: the group used by normal videos (label_bin==0) becomes class_id 0.
    All anomaly groups follow in sorted order for determinism.
    """
    normal_groups = set()
    anomaly_groups = set()

    for vid in train_video_ids:
        if vid not in video_labels:
            raise ValueError(f"Train video missing in labels index: {vid}")

        lab = video_labels[vid]
        group = str(lab.group)

        if int(lab.label_bin) == 0:
            normal_groups.add(group)
        else:
            anomaly_groups.add(group)

    if len(normal_groups) == 0:
        raise ValueError("No normal groups found (label_bin==0). Check videos.jsonl.")

    if len(normal_groups) > 1:
        # This is unusual and indicates inconsistent labeling of normal videos.
        # We fail loudly so you can fix the dataset rather than training on ambiguous taxonomy.
        raise ValueError(f"Multiple normal group names detected: {sorted(normal_groups)}")

    normal_group = list(normal_groups)[0]

    ordered = [normal_group] + sorted(anomaly_groups)
    label_map = {g: i for i, g in enumerate(ordered)}
    return label_map


def save_label_map(label_map: Dict[str, int], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(label_map, indent=2))


def load_label_map(path: str) -> Dict[str, int]:
    return json.loads(Path(path).read_text())
