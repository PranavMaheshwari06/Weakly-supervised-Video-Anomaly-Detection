"""Paired video-level bootstrap CIs for three reported test quantities.

Reads only the canonical package and the dense per-video records it lists.
Resamples test videos with replacement (same indices for all three seeds and
for learned vs reference), so the intervals describe video-sampling
uncertainty only; seed variance is reported separately as the sample SD.
Settings are fixed in advance: B=10000, RNG seed 0, 95% percentile intervals.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PACKAGE = Path("artifacts/stage2/evidence/canonical_thesis_v1/package.json")
OUTPUT = Path("artifacts/stage2/evidence/bootstrap_v1/bootstrap_cis.json")
RUNS = ("dispatch_repro_s1337", "dispatch_repro_s2027", "dispatch_repro_s3407")
B, SEED, LEVEL = 10000, 0, 0.95


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_f1(target: np.ndarray, pred: np.ndarray) -> float:
    tp = (target & pred).sum(0)
    fp = (~target & pred).sum(0)
    fn = (target & ~pred).sum(0)
    denom = 2 * tp + fp + fn
    return float(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean())


def fixed_bag_arrays(package):
    records = package["phase1_fixed_bag_dispatch"]["per_video_records"]
    ids = [r["video_id"] for r in records[RUNS[0]]]
    arrays = []
    for run in RUNS:
        rows = records[run]
        if [r["video_id"] for r in rows] != ids:
            raise ValueError(f"{run}: video order differs from {RUNS[0]}")
        arrays.append({
            "y": np.array([r["binary_true"] for r in rows]),
            "score": np.array([r["binary_score"] for r in rows]),
            "gate": np.array([r["binary_pred"] for r in rows], dtype=bool),
            "target": np.array([r["dispatch_target"] for r in rows], dtype=bool),
            "gated": np.array([r["gated"] for r in rows], dtype=bool),
        })
    return ids, arrays


def temporal_arrays(package):
    listed = package["phase5_temporal_system"]["record_files"]
    per_run = {run: {} for run in RUNS}
    for entry in listed:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        learned = record["temporal_metrics"]["primary"]["ap"]
        if record["annotation_status"] == "anomaly" and learned is not None:
            per_run[record["checkpoint"]][record["video_id"]] = (
                learned, record["baseline_metrics"]["whole_video"]["ap"])
    ids = sorted(per_run[RUNS[0]])
    if any(sorted(per_run[run]) != ids for run in RUNS):
        raise ValueError("eligible temporal videos differ between seeds")
    learned = np.array([[per_run[run][v][0] for v in ids] for run in RUNS])
    whole = np.array([[per_run[run][v][1] for v in ids] for run in RUNS])
    return ids, learned, whole


def fixed_stats(arrays, idx):
    auroc, margin = [], []
    for a in arrays:
        auroc.append(roc_auc_score(a["y"][idx], a["score"][idx]))
        all_services = np.repeat(a["gate"][idx, None], 3, axis=1)
        margin.append(macro_f1(a["target"][idx], a["gated"][idx])
                      - macro_f1(a["target"][idx], all_services))
    return float(np.mean(auroc)), float(np.mean(margin))


def interval(values):
    tail = (1 - LEVEL) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return [float(low), float(high)]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    package = json.loads(PACKAGE.read_text(encoding="utf-8"))
    fixed_ids, arrays = fixed_bag_arrays(package)
    temporal_ids, learned, whole = temporal_arrays(package)

    n_fixed, n_temporal = len(fixed_ids), len(temporal_ids)
    auroc, margin = fixed_stats(arrays, np.arange(n_fixed))
    ap_gap = float((learned - whole).mean(1).mean())
    # Point estimates must reproduce the values reported in the thesis.
    assert n_fixed == 290 and n_temporal == 135
    assert round(auroc, 3) == 0.940, auroc
    assert round(margin, 3) == 0.015, margin
    assert round(float(learned.mean(1).mean()), 3) == 0.327
    assert round(float(whole.mean(1).mean()), 3) == 0.191

    rng = np.random.default_rng(SEED)
    boot_auroc, boot_margin, boot_ap = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n_fixed, n_fixed)
        # AUROC needs both classes; a resample without them is redrawn.
        while len(np.unique(arrays[0]["y"][idx])) < 2:
            idx = rng.integers(0, n_fixed, n_fixed)
        a, m = fixed_stats(arrays, idx)
        boot_auroc.append(a); boot_margin.append(m)
        t_idx = rng.integers(0, n_temporal, n_temporal)
        boot_ap.append(float((learned[:, t_idx] - whole[:, t_idx]).mean(1).mean()))

    result = {
        "scope": "paired video-level bootstrap over the fixed test population; "
                 "video-sampling uncertainty only, not training/seed variance",
        "settings": {"resamples": B, "rng_seed": SEED, "level": LEVEL, "interval": "percentile",
                     "statistic": "mean over seeds 1337/2027/3407 per resample"},
        "inputs": {"package": str(PACKAGE), "package_sha256": sha256(PACKAGE),
                   "temporal_record_files": len(package["phase5_temporal_system"]["record_files"])},
        "fixed_bag_binary_auroc": {"videos": n_fixed, "point": auroc, "ci": interval(boot_auroc)},
        "fixed_bag_gated_margin_macro_f1": {
            "definition": "learned binary-gated Macro-F1 minus binary-gated all-services Macro-F1",
            "videos": n_fixed, "point": margin, "ci": interval(boot_margin),
            "share_resamples_le_zero": float(np.mean(np.array(boot_margin) <= 0))},
        "dense_temporal_ap_minus_whole_video": {
            "videos": n_temporal, "point": ap_gap, "ci": interval(boot_ap)},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
