#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts/stage2/evidence/teacher_ablation_v1/summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def binary_auroc(targets: list[int], scores: list[float]) -> float:
    ordered = sorted(zip(scores, targets))
    rank_sum_positive = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum_positive += average_rank * sum(target for _, target in ordered[index:end])
        index = end
    positives = sum(targets)
    negatives = len(targets) - positives
    return (rank_sum_positive - positives * (positives + 1) / 2) / (positives * negatives)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, message: str) -> None:
    expect(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), message)


def metrics(rows: list[dict]) -> dict[str, float]:
    targets = [row["dispatch_target"] for row in rows]
    preds = [row["dispatch_pred"] for row in rows]
    channel_f1 = []
    for channel in range(3):
        tp = sum(t[channel] == p[channel] == 1 for t, p in zip(targets, preds))
        fp = sum(t[channel] == 0 and p[channel] == 1 for t, p in zip(targets, preds))
        fn = sum(t[channel] == 1 and p[channel] == 0 for t, p in zip(targets, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        channel_f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    binary_targets = [row["binary_true"] for row in rows]
    binary_preds = [row["binary_pred"] for row in rows]
    tp = sum(t == p == 1 for t, p in zip(binary_targets, binary_preds))
    fp = sum(t == 0 and p == 1 for t, p in zip(binary_targets, binary_preds))
    fn = sum(t == 1 and p == 0 for t, p in zip(binary_targets, binary_preds))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "macro_f1": statistics.mean(channel_f1),
        "binary_auroc": binary_auroc(binary_targets, [row["binary_score"] for row in rows]),
        "binary_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "hamming_loss": sum(
            a != b for target, pred in zip(targets, preds) for a, b in zip(target, pred)
        ) / (len(rows) * 3),
    }


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    expect(summary["evidence_id"] == "teacher_ablation_v1", "evidence ID")
    expect(summary["seeds"] == [1337, 2027, 3407], "seed set/order")
    expect(len(summary["runs"]) == 12, "run count")
    expect(summary["failed_runs"] == [] and summary["reruns"] == [], "failure/rerun record")

    for path, expected_hash in summary["implementation_hashes"].items():
        expect(sha256_file(ROOT / path) == expected_hash, f"implementation hash: {path}")
    for info in summary["configs"].values():
        expect(sha256_file(ROOT / info["path"]) == info["sha256"], f"config hash: {info['path']}")
    for path, expected_hash in summary["protected_test_preflight_hashes_not_loaded_by_runs"].items():
        expect(sha256_file(ROOT / path) == expected_hash, f"protected test hash: {path}")
    fixed_paths = {
        "train_split_sha256": "data/ucf_crime/splits/dman_v21/train.txt",
        "validation_split_sha256": "data/ucf_crime/splits/dman_v21/val.txt",
        "validation_bag_manifest_sha256": "artifacts/stage2/bag_manifests/dispatch_lora_fixed_val.json",
        "teacher_train_sha256": "artifacts/stage2/teacher/teacher_scores_train.pt",
        "teacher_validation_sha256": "artifacts/stage2/teacher/teacher_scores_val.pt",
    }
    for key, path in fixed_paths.items():
        expect(sha256_file(ROOT / path) == summary["fixed_inputs"][key], f"fixed input: {path}")

    values: dict[str, dict[str, list[float]]] = {}
    plan_files: dict[tuple[str, int], str] = {}
    for run in summary["runs"]:
        for path, expected_hash in run["artifact_hashes"].items():
            expect(sha256_file(ROOT / path) == expected_hash, f"run hash: {path}")
        evaluation_path = next(path for path in run["artifact_hashes"] if path.endswith("dispatch_eval_val.json"))
        evaluation = json.loads((ROOT / evaluation_path).read_text(encoding="utf-8"))
        expect(len(evaluation["per_video"]) == 161, f"{run['run_tag']}: validation count")
        expect(evaluation["provenance"]["threshold_policy"] == {
            "dispatch_primary": 0.5,
            "mode": "fixed",
            "selection_procedure": None,
        }, f"{run['run_tag']}: threshold policy")
        calculated = metrics(evaluation["per_video"])
        for name, value in calculated.items():
            close(value, run["metrics"][name], f"{run['run_tag']}: {name}")
            values.setdefault(run["variant"], {}).setdefault(name, []).append(value)
        manifest_path = next(path for path in run["artifact_hashes"] if path.endswith("run_manifest.json"))
        manifest = json.loads((ROOT / manifest_path).read_text(encoding="utf-8"))
        expect(manifest["validation_only"] is True, f"{run['run_tag']}: validation-only")
        expect(set(manifest["input_paths"]["teacher_scores"]) == {"train", "val"},
               f"{run['run_tag']}: teacher embargo")
        expect(set(manifest["input_paths"]["evaluation_bag_manifests"]) == {"val"},
               f"{run['run_tag']}: manifest embargo")
        expect(set(manifest["artifact_hashes"]["feature_store"]) == {"train", "val"},
               f"{run['run_tag']}: feature-store embargo")
        expect(not any("test" in path.lower() for path in manifest["artifact_hashes"]["files"]),
               f"{run['run_tag']}: test file hash in run manifest")
        plan_path = next(path for path in run["artifact_hashes"] if path.endswith("training_bag_plan.json"))
        plan = json.loads((ROOT / plan_path).read_text(encoding="utf-8"))
        calculated_plan_identity = sha256_json({
            "schema_version": plan["schema_version"],
            "metadata": plan["metadata"],
            "epoch_identities": plan["epoch_identities"],
        })
        expect(calculated_plan_identity == plan["plan_identity_sha256"], f"{run['run_tag']}: plan")
        plan_files[(run["variant"], run["seed"])] = sha256_file(ROOT / plan_path)

    for seed in summary["seeds"]:
        expect(plan_files[("full", seed)] == plan_files[("no_input", seed)], f"seed {seed}: guided pair")
        expect(plan_files[("uniform", seed)] == plan_files[("neither", seed)], f"seed {seed}: uniform pair")

    for variant, metric_values in values.items():
        for name, observed in metric_values.items():
            expected = summary["aggregates"][variant][name]
            close(statistics.mean(observed), expected["mean"], f"{variant}: {name} mean")
            close(statistics.stdev(observed), expected["sample_std"], f"{variant}: {name} std")

    print(
        "PASS: verified 12 validation-only runs, 84 run artifacts, independently "
        "recomputed 48 metrics, four aggregates, and six paired bag-plan identities"
    )


if __name__ == "__main__":
    main()
