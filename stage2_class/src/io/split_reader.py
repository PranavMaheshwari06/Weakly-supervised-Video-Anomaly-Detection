from __future__ import annotations

from pathlib import Path
from typing import List


def read_split_txt(split_path: str) -> List[str]:
    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    video_ids: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        video_ids.append(line)

    if len(video_ids) == 0:
        raise ValueError(f"Split file is empty: {path}")

    return video_ids
