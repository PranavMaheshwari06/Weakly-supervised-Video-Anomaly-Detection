#!/usr/bin/env python3
"""Verify the retained Phase-2 historical 14-class evidence without model inference."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reproducibility/evidence/historical_14_class_v1/summary.json"
PHASE1_INVENTORY = ROOT / "reproducibility/evidence/ARTIFACT_SHA256_INVENTORY.json"
TOL = 1e-12


def fail(message: str) -> None:
    raise AssertionError(message)


def expect(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def expect_close(actual: float, expected: float, label: str, tolerance: float = TOL) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def load_json(relative_path: str | Path) -> Any:
    return json.loads((ROOT / relative_path).read_text())


def sha256(relative_path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(relative_path: str, expected_hash: str) -> None:
    expect((ROOT / relative_path).is_file(), f"missing evidence file: {relative_path}")
    expect(sha256(relative_path) == expected_hash, f"SHA-256 mismatch: {relative_path}")


def float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def macro_metrics(targets: list[int], predictions: list[int], classes: int) -> tuple[float, float]:
    recalls: list[float] = []
    f1_values: list[float] = []
    for class_id in range(classes):
        tp = sum(t == class_id and p == class_id for t, p in zip(targets, predictions))
        fp = sum(t != class_id and p == class_id for t, p in zip(targets, predictions))
        fn = sum(t == class_id and p != class_id for t, p in zip(targets, predictions))
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1_value = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_values.append(f1_value)
    return sum(recalls) / classes, sum(f1_values) / classes


def confusion_matrix(targets: list[int], predictions: list[int], classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1
    return matrix


def binary_metrics(targets: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    predictions = [int(float32(score) >= float(threshold)) for score in scores]
    tp = sum(t == 1 and p == 1 for t, p in zip(targets, predictions))
    fp = sum(t == 0 and p == 1 for t, p in zip(targets, predictions))
    fn = sum(t == 1 and p == 0 for t, p in zip(targets, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_value = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1_value}


def auroc(targets: list[int], scores: list[float]) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    ranked = sorted((float32(score), target) for target, score in zip(targets, scores))
    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    expect(positive_count > 0 and negative_count > 0, "AUROC requires both binary classes")
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][0] == ranked[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(target for _, target in ranked[start:end])
        start = end
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )


def best_f1_threshold(targets: list[int], scores: list[float]) -> dict[str, float]:
    thresholds = sorted({float32(score) for score in scores})
    best = {"threshold": thresholds[0], "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in thresholds:
        current = {"threshold": threshold, **binary_metrics(targets, scores, threshold)}
        if (
            current["f1"] > best["f1"]
            or (abs(current["f1"] - best["f1"]) < TOL and current["recall"] > best["recall"])
            or (
                abs(current["f1"] - best["f1"]) < TOL
                and abs(current["recall"] - best["recall"]) < TOL
                and current["threshold"] < best["threshold"]
            )
        ):
            best = current
    return best


def load_video_index(path: str) -> dict[str, dict[str, Any]]:
    records = [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]
    return {record["rel_path"]: record for record in records}


def check_hashes(summary: dict[str, Any]) -> int:
    files: dict[str, str] = {}
    files.update(summary["implementation_snapshot"]["files"])
    for split in summary["task"]["split_files"].values():
        files[split["path"]] = split["sha256"]
    videos_index = summary["task"]["videos_index"]
    files[videos_index["path"]] = videos_index["sha256"]
    for teacher_key in ("frozen_teacher_scores", "lora_teacher_scores"):
        teacher = summary["inputs"][teacher_key]
        files.update(teacher["files"])
        files[teacher["prototype"]["path"]] = teacher["prototype"]["sha256"]
    for experiment in summary["experiments"].values():
        files.update(experiment["files"])
    for run in summary["negative_runs"]["runs"].values():
        for relative_path, expected_hash in run.values():
            files[relative_path] = expected_hash
    for source in summary["composite_row_source_trace"].values():
        if not isinstance(source, dict):
            continue
        for key in ("evaluation_file", "checkpoint", "config"):
            if key in source:
                relative_path, expected_hash = source[key]
                files[relative_path] = expected_hash
    for relative_path, expected_hash in sorted(files.items()):
        check_file(relative_path, expected_hash)
    return len(files)


def check_feature_identities(summary: dict[str, Any]) -> None:
    inventory = load_json(PHASE1_INVENTORY.relative_to(ROOT))
    indexed = {
        (item["root"], item["split"]): item["identity_sha256"]
        for item in inventory["feature_stores"]
    }
    for source_key in ("frozen_feature_store", "lora_feature_store"):
        source = summary["inputs"][source_key]
        for split, identity in source["split_identities"].items():
            expect(indexed.get((source["path"], split)) == identity, f"feature identity mismatch: {source_key}/{split}")


def check_external_evaluation(
    summary: dict[str, Any], experiment_id: str, video_index: dict[str, dict[str, Any]]
) -> None:
    experiment = summary["experiments"][experiment_id]
    class_names = summary["task"]["class_names"]
    class_to_id = {name: index for index, name in enumerate(class_names)}
    eval_root = Path("artifacts/stage2/eval") / experiment_id

    label_map = load_json(Path("artifacts/stage2/runs") / experiment_id / "label_map/label_map.json")
    expect(label_map == class_to_id, f"{experiment_id}: label map/order mismatch")

    split_payloads: dict[str, dict[str, Any]] = {}
    binary_targets_by_split: dict[str, list[int]] = {}
    for split in ("val", "test"):
        payload = load_json(eval_root / f"phase2_{split}.json")
        split_payloads[split] = payload
        expected = experiment[f"external_{split}"]
        per_video = payload["per_video"]
        split_ids = [line.strip() for line in (ROOT / summary["task"]["split_files"][split]["path"]).read_text().splitlines() if line.strip()]

        expect(payload["meta"]["class_names"] == class_names, f"{experiment_id}/{split}: class names")
        expect(payload["meta"]["split"] == split, f"{experiment_id}/{split}: split metadata")
        expect(payload["meta"]["run_tag"] == experiment_id, f"{experiment_id}/{split}: run tag")
        expect(payload["meta"]["cfg_path"] == experiment["config"], f"{experiment_id}/{split}: config path")
        expect(payload["meta"]["trained_epoch"] == experiment["selected_epoch"], f"{experiment_id}/{split}: epoch")
        expect(payload["meta"]["sampler_cfg"]["bag_size"] == 16, f"{experiment_id}/{split}: bag size")
        expect(per_video["video_ids"] == split_ids, f"{experiment_id}/{split}: video order differs from split")

        lengths = {len(values) for values in per_video.values()}
        expect(lengths == {len(split_ids)}, f"{experiment_id}/{split}: per-video arrays differ in length")
        targets = [int(value) for value in per_video["targets"]]
        predictions = [int(value) for value in per_video["preds"]]
        expected_targets = [
            0 if int(video_index[video_id]["label_bin"]) == 0 else class_to_id[video_index[video_id]["group"]]
            for video_id in split_ids
        ]
        expect(targets == expected_targets, f"{experiment_id}/{split}: targets disagree with video index")
        expect(all(0 <= value < 14 for value in predictions), f"{experiment_id}/{split}: invalid prediction")

        matrix = confusion_matrix(targets, predictions, 14)
        recall, f1_value = macro_metrics(targets, predictions, 14)
        accuracy = sum(t == p for t, p in zip(targets, predictions)) / len(targets)
        distribution = {str(key): value for key, value in sorted(Counter(predictions).items())}
        expect(matrix == payload["confusion_matrix"], f"{experiment_id}/{split}: confusion matrix")
        expect(distribution == {str(k): v for k, v in payload["metrics"]["pred_class_dist"].items()}, f"{experiment_id}/{split}: stored prediction distribution")
        expect(distribution == expected["prediction_distribution"], f"{experiment_id}/{split}: summary prediction distribution")
        expect_close(accuracy, payload["metrics"]["accuracy"], f"{experiment_id}/{split}: stored accuracy")
        expect_close(recall, payload["metrics"]["macro_recall"], f"{experiment_id}/{split}: stored Macro-Recall")
        expect_close(f1_value, payload["metrics"]["macro_f1"], f"{experiment_id}/{split}: stored Macro-F1")
        expect_close(accuracy, expected["accuracy"], f"{experiment_id}/{split}: summary accuracy")
        expect_close(recall, expected["macro_recall"], f"{experiment_id}/{split}: summary Macro-Recall")
        expect_close(f1_value, expected["macro_f1"], f"{experiment_id}/{split}: summary Macro-F1")
        expect(len(targets) == expected["count_videos"], f"{experiment_id}/{split}: count")
        binary_targets_by_split[split] = [int(video_index[video_id]["label_bin"]) for video_id in split_ids]

    sidecar = load_json(eval_root / "phase2_confusion_test.json")
    expect(sidecar["class_names"] == class_names, f"{experiment_id}: confusion sidecar labels")
    expect(sidecar["matrix"] == split_payloads["test"]["confusion_matrix"], f"{experiment_id}: confusion sidecar")

    threshold_record = load_json(eval_root / "phase2_binary_threshold.json")
    threshold = float(threshold_record["threshold"])
    expect(threshold_record["selected_on"] == "val" and threshold_record["policy"] == "best_f1", f"{experiment_id}: threshold policy")
    selected = best_f1_threshold(binary_targets_by_split["val"], split_payloads["val"]["per_video"]["prob_anomaly"])
    expect_close(selected["threshold"], threshold, f"{experiment_id}: selected threshold")

    for split in ("val", "test"):
        binary_record = load_json(eval_root / f"phase2_binary_{split}.json")
        scores = split_payloads[split]["per_video"]["prob_anomaly"]
        calculated = binary_metrics(binary_targets_by_split[split], scores, threshold)
        calculated["auroc"] = auroc(binary_targets_by_split[split], scores)
        for metric in ("precision", "recall", "f1", "auroc"):
            expect_close(calculated[metric], binary_record[metric], f"{experiment_id}/{split}: binary {metric}")
        expect(binary_record["count_videos"] == len(scores), f"{experiment_id}/{split}: binary count")
        if split == "test":
            for metric in ("precision", "recall", "f1", "auroc"):
                expect_close(calculated[metric], experiment["external_test"]["binary"][metric], f"{experiment_id}: summary binary {metric}")

    run_summary = load_json(Path("artifacts/stage2/runs") / experiment_id / "train/m27_val_metrics.json")
    expected_best = experiment["checkpoint_selection_validation"]["macro_f1"]
    expect_close(run_summary["best_val_macro_f1"], expected_best, f"{experiment_id}: run summary best validation")
    log_path = ROOT / "artifacts/stage2/runs" / experiment_id / "train/m27_train_log.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    selected_records = [record for record in records if record["epoch"] == experiment["selected_epoch"]]
    expect(len(selected_records) == 1, f"{experiment_id}: selected epoch record")
    expect_close(selected_records[0]["val_macro_f1"], expected_best, f"{experiment_id}: selected validation metric")


def check_negative_runs(summary: dict[str, Any]) -> None:
    expected_f1 = 0.04742145820983995
    for run_tag in summary["negative_runs"]["runs"]:
        run_root = ROOT / "artifacts/stage2/runs" / run_tag
        run_summary = json.loads((run_root / "train/m27_val_metrics.json").read_text())
        records = [json.loads(line) for line in (run_root / "train/m27_train_log.jsonl").read_text().splitlines() if line.strip()]
        label_map = json.loads((run_root / "label_map/label_map.json").read_text())
        expect(len(records) == 9 and run_summary["epochs_ran"] == 9, f"{run_tag}: expected nine epochs")
        expect_close(run_summary["best_val_macro_f1"], expected_f1, f"{run_tag}: summary validation Macro-F1")
        expect(label_map.get("Normal") == 0 and len(label_map) == 14, f"{run_tag}: label map")
        for record in records:
            expect_close(record["val_macro_f1"], expected_f1, f"{run_tag}/epoch {record['epoch']}: Macro-F1")
            expect_close(record["val_macro_recall"], 1 / 14, f"{run_tag}/epoch {record['epoch']}: Macro-Recall")
            expect_close(record["val_accuracy"], 80 / 161, f"{run_tag}/epoch {record['epoch']}: accuracy")
            expect(record["val_pred_class_dist"] == {"0": 161}, f"{run_tag}/epoch {record['epoch']}: predictions")


def check_composite_row_sources(summary: dict[str, Any]) -> None:
    trace = summary["composite_row_source_trace"]
    for trace_key in ("e1_multiclass_values", "e2_recall_and_accuracy_values"):
        source = trace[trace_key]
        payload = load_json(source["evaluation_file"][0])
        expect(payload["meta"]["run_tag"] == source["actual_run_tag"], f"{trace_key}: run tag")
        for name, expected in source["values"].items():
            metric_name = name.removeprefix("test_")
            expect_close(payload["metrics"][metric_name], expected, f"{trace_key}: {metric_name}")


def check_implementation_distinction() -> None:
    evaluator = (ROOT / "stage2_class/src/eval/m27_eval_hier_split.py").read_text()
    trainer = (ROOT / "stage2_class/src/train/m27_train_hier_mil.py").read_text()
    expect("pred = int(torch.argmax(full_probs, dim=1).item())" in evaluator, "external decision rule changed")
    expect("epoch=0," in evaluator, "external epoch-0 bag protocol changed")
    expect("bin_preds = torch.argmax(binary_logits, dim=1)" in trainer, "training cascaded decision rule changed")
    expect("full_preds = torch.where(bin_preds == 1" in trainer, "training binary gate changed")


def main() -> int:
    try:
        summary = load_json(SUMMARY_PATH.relative_to(ROOT))
        expect(summary["evidence_id"] == "historical_14_class_v1", "wrong evidence summary")
        file_count = check_hashes(summary)
        check_feature_identities(summary)
        video_index = load_video_index(summary["task"]["videos_index"]["path"])
        for experiment_id in ("hier_cascaded_v3", "hier_lora_e2e"):
            check_external_evaluation(summary, experiment_id, video_index)
        check_negative_runs(summary)
        check_composite_row_sources(summary)
        check_implementation_distinction()
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "PASS: verified "
        f"{file_count} hashed files, 6 feature-store identities, "
        "4 direct evaluations, 4 binary evaluations, 2 confusion sidecars, "
        "and 4 validation-only negative runs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
