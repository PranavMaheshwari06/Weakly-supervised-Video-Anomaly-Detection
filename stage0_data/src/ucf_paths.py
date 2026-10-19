from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class VideoRecord:
    video_id: str      # e.g., "Abuse028_x264"
    rel_path: str      # e.g., "Abuse/Abuse028_x264.mp4"
    abs_path: str      # absolute file path
    label: int         # 0 = normal, 1..13 = anomaly classes
    label_name: str    # folder name under raw/


DEFAULT_UCF_ROOT_REL = Path("data") / "ucf_crime"
DEFAULT_RAW_REL = DEFAULT_UCF_ROOT_REL / "raw"
TRAIN_NORMAL_FOLDER = "Training_Normal_Videos_Anomaly"
TEST_NORMAL_FOLDER = "Testing_Normal_Videos_Anomaly"

ANOMALY_CLASS_FOLDERS = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]


def build_label_map() -> Dict[str, int]:
    """
    Label convention:
      - both normal folders map to 0
      - anomaly classes map to 1..13 in fixed order
    """
    label_map: Dict[str, int] = {
        TRAIN_NORMAL_FOLDER: 0,
        TEST_NORMAL_FOLDER: 0,
    }
    for idx, name in enumerate(ANOMALY_CLASS_FOLDERS, start=1):
        label_map[name] = idx
    return label_map


def required_raw_folders() -> List[str]:
    return [TRAIN_NORMAL_FOLDER, TEST_NORMAL_FOLDER] + ANOMALY_CLASS_FOLDERS


def assert_raw_layout(raw_root: Path) -> None:
    """
    Hard fail if the dataset layout is not what Stage 0 expects.
    """
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root does not exist or is not a directory: {raw_root}")

    missing = [name for name in required_raw_folders() if not (raw_root / name).is_dir()]
    if missing:
        raise RuntimeError(
            "Missing required folders under raw/. Missing:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + f"\nraw_root={raw_root}"
        )



def index_all_videos(raw_root: Path) -> List[VideoRecord]:
    """
    Index all .mp4 files under the required folders (non-recursive).
    Returns deterministic ordering.
    """
    assert_raw_layout(raw_root)
    label_map = build_label_map()

    records: List[VideoRecord] = []

    for folder_name in required_raw_folders():
        folder_path = raw_root / folder_name
        mp4_files = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])

        if not mp4_files:
            raise RuntimeError(f"No .mp4 files found in: {folder_path}")

        for mp4_path in mp4_files:
            video_id = mp4_path.stem
            rel_path = f"{folder_name}/{mp4_path.name}"
            records.append(
                VideoRecord(
                    video_id=video_id,
                    rel_path=rel_path,
                    abs_path=str(mp4_path.resolve()),
                    label=label_map[folder_name],
                    label_name=folder_name,
                )
            )

    # Stable ordering helps reproducibility and debugging
    records.sort(key=lambda r: (r.label_name, r.video_id))
    return records
