"""
Stage 0 - Split Builder (UCF-Crime Anomaly Detection, DMAN v2.1)

Reads the official anomaly-detection split lists:
  data/ucf_crime/splits/Anomaly_Detection_splits/Anomaly_Train.txt
  data/ucf_crime/splits/Anomaly_Detection_splits/Anomaly_Test.txt

Validates that each referenced video exists under:
  data/ucf_crime/raw/

Creates a deterministic validation split from the official train list,
stratified by top-level folder name (e.g., Abuse, Arrest, Training_Normal_Videos_Anomaly).

Writes:
  data/ucf_crime/splits/dman_v21/train.txt
  data/ucf_crime/splits/dman_v21/val.txt
  data/ucf_crime/splits/dman_v21/test.txt
  data/ucf_crime/splits/dman_v21/stats.json
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def read_lines(txt_path: Path) -> list[str]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing split file: {txt_path}")
    lines = []
    for line in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(line.replace("\\", "/"))
    if not lines:
        raise ValueError(f"Split file is empty: {txt_path}")
    return lines


def resolve_to_relpaths(lines: list[str], raw_root: Path) -> list[str]:
    """
    Official lines are usually like: Abuse/Abuse028_x264.mp4
    We keep them as relative paths and verify they exist under raw_root.
    """
    resolved = []
    missing = []

    for entry in lines:
        rel = entry.lstrip("./")
        abs_path = raw_root / rel
        if not abs_path.exists():
            missing.append(entry)
        else:
            resolved.append(rel)

    if missing:
        raise FileNotFoundError(
            "Some split entries do not exist under raw/. "
            f"Missing count={len(missing)}. Example: {missing[:10]}"
        )

    # deterministic ordering
    return sorted(set(resolved))


def top_group(rel_path: str) -> str:
    # first folder name: Abuse/..., Training_Normal_Videos_Anomaly/...
    return Path(rel_path).parts[0]


def make_val_split(
    train_relpaths: list[str],
    val_ratio: float,
    seed: int,
    min_val_per_group: int = 1,
) -> tuple[list[str], list[str]]:
    """
    Deterministic val split:
    - group by top folder
    - sample approx val_ratio per group
    - ensure at least min_val_per_group when group has enough samples
    """
    rng = random.Random(seed)

    by_group = defaultdict(list)
    for p in train_relpaths:
        by_group[top_group(p)].append(p)

    val_set = set()

    for group, items in by_group.items():
        items = sorted(items)
        n = len(items)

        if n <= 1:
            continue  # can't take val safely

        target = int(round(n * val_ratio))
        target = max(target, min_val_per_group)
        target = min(target, n - 1)  # keep at least 1 in train

        items_shuffled = items[:]
        rng.shuffle(items_shuffled)

        for chosen in items_shuffled[:target]:
            val_set.add(chosen)

    val = sorted(val_set)
    train = sorted([p for p in train_relpaths if p not in val_set])

    if len(train) == 0:
        raise RuntimeError("Val split consumed all training samples. Lower val_ratio.")

    return train, val


def write_txt(path: Path, relpaths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(relpaths) + "\n", encoding="utf-8")


def count_by_group(items: list[str]) -> dict[str, int]:
    counts = defaultdict(int)
    for p in items:
        counts[top_group(p)] += 1
    return dict(sorted(counts.items(), key=lambda x: x[0]))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]


    raw_root = repo_root / "data" / "ucf_crime" / "raw"
    official_dir = repo_root / "data" / "ucf_crime" / "splits" / "Anomaly_Detection_splits"
    out_dir = repo_root / "data" / "ucf_crime" / "splits" / "dman_v21"

    train_txt = official_dir /"Anomaly_Train.txt"
    test_txt = official_dir /"Anomaly_Test.txt"

    train_lines = read_lines(train_txt)
    test_lines = read_lines(test_txt)

    train_all = resolve_to_relpaths(train_lines, raw_root)
    test_all = resolve_to_relpaths(test_lines, raw_root)

    overlap = set(train_all) & set(test_all)
    if overlap:
        raise RuntimeError(f"Train/Test overlap detected. Example: {sorted(list(overlap))[:10]}")

    val_ratio = 0.10
    seed = 42
    train_final, val_final = make_val_split(train_all, val_ratio=val_ratio, seed=seed, min_val_per_group=1)

    write_txt(out_dir / "train.txt", train_final)
    write_txt(out_dir / "val.txt", val_final)
    write_txt(out_dir / "test.txt", test_all)

    stats = {
        "totals": {
            "train": len(train_final),
            "val": len(val_final),
            "test": len(test_all),
            "train_plus_val": len(train_final) + len(val_final),
            "all": len(train_final) + len(val_final) + len(test_all),
        },
        "by_group": {
            "train": count_by_group(train_final),
            "val": count_by_group(val_final),
            "test": count_by_group(test_all),
        },
        "config": {"val_ratio": val_ratio, "seed": seed, "min_val_per_group": 1},
        "paths": {
            "raw_root": str(raw_root),
            "official_train": str(train_txt),
            "official_test": str(test_txt),
            "out_dir": str(out_dir),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Split builder finished")
    print("Totals:", stats["totals"])
    print("Wrote:", out_dir)


if __name__ == "__main__":
    main()
