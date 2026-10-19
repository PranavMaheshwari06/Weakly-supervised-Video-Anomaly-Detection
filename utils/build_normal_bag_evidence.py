"""Builds the small, Git-visible evidence mirror for the normal-bag response-loss
study, following the pattern of reproducibility/evidence/dispatch_repro_v1/:
aggregate mean/sample_std per arm, no checkpoints or feature stores. Validation-only; test split is never read.

Usage:
    PYTHONPATH=. python utils/build_normal_bag_evidence.py
"""
import hashlib
import json
import statistics as st
from pathlib import Path

from stage2_class.src.train.m27_train_dispatch_mil import compute_dispatch_metrics

ALL = [1, 1, 1]
ZERO = [0, 0, 0]
SEEDS = [1337, 2027, 3407]
ARMS = {
    "control": "ablation_full_s{seed}",
    "tier1a_unweighted": "abl_normal_response_v1_s{seed}",
    "tier1c_reweighted": "abl_normal_response_reweighted_v1_s{seed}",
}
OUT_DIR = Path("reproducibility/evidence/normal_bag_response_study")


def load_metrics(run_tag):
    ev = json.loads((Path("artifacts/stage2/eval") / run_tag / "dispatch_eval_val.json").read_text())
    vids = ev["per_video"]
    t = [v["dispatch_target"] for v in vids]
    p = [v["dispatch_pred"] for v in vids]
    b = [v["binary_pred"] for v in vids]
    anom = [i for i, v in enumerate(vids) if v["binary_true"] == 1]
    raw = compute_dispatch_metrics(t, p)
    gated_learned = compute_dispatch_metrics(t, [x if g else ZERO for x, g in zip(p, b)])
    gated_all = compute_dispatch_metrics(t, [ALL if g else ZERO for g in b])
    anom_learned = compute_dispatch_metrics([t[i] for i in anom], [p[i] for i in anom])
    anom_all = compute_dispatch_metrics([t[i] for i in anom], [ALL] * len(anom))
    return {
        "binary_auroc": ev["binary_auroc"],
        "binary_f1": ev["binary_f1"],
        "raw_macro_f1": raw["macro_f1"],
        "raw_police_precision": raw["police_precision"],
        "raw_police_recall": raw["police_recall"],
        "raw_medical_precision": raw["medical_precision"],
        "raw_medical_recall": raw["medical_recall"],
        "raw_fire_precision": raw["fire_precision"],
        "raw_fire_recall": raw["fire_recall"],
        "gated_margin": gated_learned["macro_f1"] - gated_all["macro_f1"],
        "anom81_margin": anom_learned["macro_f1"] - anom_all["macro_f1"],
    }


def agg(values):
    return {"mean": st.mean(values), "sample_std": st.stdev(values) if len(values) > 1 else 0.0}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "validation_only_never_test",
        "n_validation_videos": 161,
        "seeds": SEEDS,
        "training_bag_plan_identity": "shared across all three arms per-seed (see REPORT.md)",
        "arms": {},
    }
    for arm, pattern in ARMS.items():
        per_seed = {seed: load_metrics(pattern.format(seed=seed)) for seed in SEEDS}
        keys = next(iter(per_seed.values())).keys()
        summary["arms"][arm] = {
            "run_tags": [pattern.format(seed=s) for s in SEEDS],
            "per_seed": per_seed,
            "aggregate": {k: agg([per_seed[s][k] for s in SEEDS]) for k in keys},
        }

    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"Wrote {out_path} sha256={digest}")


if __name__ == "__main__":
    main()
