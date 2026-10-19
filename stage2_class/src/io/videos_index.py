from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class VideoLabel:
    video_id: str
    label_bin: int
    group: str


def load_videos_index(jsonl_path: str) -> Dict[str, VideoLabel]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"videos.jsonl not found: {path}")

    out: Dict[str, VideoLabel] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            video_id = obj.get("video_id") or obj.get("rel_path") or obj.get("path")
            if video_id is None:
                raise ValueError(f"Line {line_num}: missing video_id field")

            label_bin = int(obj["label_bin"])
            group = str(obj["group"])

            if label_bin not in (0, 1):
                raise ValueError(f"Line {line_num}: label_bin must be 0/1, got {label_bin}")

            out[video_id] = VideoLabel(video_id=video_id, label_bin=label_bin, group=group)

    if len(out) == 0:
        raise ValueError(f"No records found in: {path}")

    return out
