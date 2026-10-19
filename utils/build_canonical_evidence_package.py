"""Build the immutable Phase-6 canonical quantitative evidence package.

Reads verified Phase 1/2/3/5 artifacts and saved evaluation records only. It
does not load a model, run inference, select checkpoints, or tune thresholds.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

OUTPUT = Path("artifacts/stage2/evidence/canonical_thesis_v1/package.json")
DISPATCH = Path("artifacts/stage2/evidence/dispatch_repro_v1/summary.json")
HISTORICAL = Path("reproducibility/evidence/historical_14_class_v1/summary.json")
ABLATION = Path("artifacts/stage2/evidence/teacher_ablation_v1/summary.json")
TEMPORAL = Path("artifacts/stage2/evidence/temporal_system_v1/summary.json")
RUNS = ("dispatch_repro_s1337", "dispatch_repro_s2027", "dispatch_repro_s3407")
CHANNELS = ("police", "medical", "fire")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_stats(values: list[float | int]) -> dict:
    if len(values) != 3:
        raise ValueError(f"Expected three genuine replicates, got {len(values)}")
    return {"values": values, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values), "n": 3}


def _roc_auc(y_true: list[int], y_score: list[float]) -> float:
    """Tie-aware Mann-Whitney AUROC without an ML dependency."""
    if len(y_true) != len(y_score):
        raise ValueError("AUROC inputs differ in length")
    positives, negatives = sum(y_true), len(y_true) - sum(y_true)
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    ordered = sorted(zip(y_score, y_true))
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def pr_curve(labels: list[int], scores: list[float]) -> dict:
    positives = sum(labels)
    if not positives:
        raise ValueError("PR curve requires positive examples")
    ordered = sorted(zip(scores, labels), reverse=True)
    thresholds, precision, recall = [], [1.0], [0.0]
    tp = fp = 0
    ap = previous_recall = 0.0
    start = 0
    while start < len(ordered):
        threshold = ordered[start][0]
        end = start
        while end < len(ordered) and ordered[end][0] == threshold:
            tp += ordered[end][1] == 1
            fp += ordered[end][1] == 0
            end += 1
        p, r = tp / (tp + fp), tp / positives
        thresholds.append(threshold)
        precision.append(p)
        recall.append(r)
        ap += (r - previous_recall) * p
        previous_recall = r
        start = end
    return {"thresholds": thresholds, "precision": precision, "recall": recall,
            "average_precision": ap, "positive_support": positives, "population": len(labels)}


def channel_metrics(targets: list[list[int]], predictions: list[list[int]]) -> dict:
    output = {}
    for index, channel in enumerate(CHANNELS):
        tp = fp = fn = tn = 0
        for target, prediction in zip(targets, predictions):
            truth, pred = target[index], prediction[index]
            tp += truth == 1 and pred == 1
            fp += truth == 0 and pred == 1
            fn += truth == 1 and pred == 0
            tn += truth == 0 and pred == 0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output[channel] = {"precision": precision, "recall": recall, "f1": f1,
                           "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "support_pos": tp + fn, "support_neg": tn + fp}
    return output


def routing_metrics(rows: list[dict], field: str) -> dict:
    targets, predictions = [r["dispatch_target"] for r in rows], [r[field] for r in rows]
    channels = channel_metrics(targets, predictions)
    errors = sum(a != b for target, pred in zip(targets, predictions) for a, b in zip(target, pred))
    distribution: dict[str, int] = {}
    for prediction in predictions:
        key = ",".join(map(str, prediction))
        distribution[key] = distribution.get(key, 0) + 1
    return {
        "videos": len(rows),
        "macro_f1": statistics.mean(channels[c]["f1"] for c in CHANNELS),
        "macro_recall": statistics.mean(channels[c]["recall"] for c in CHANNELS),
        "hamming_loss": errors / (len(rows) * 3),
        "exact_match_ratio": sum(a == b for a, b in zip(targets, predictions)) / len(rows),
        "per_channel": channels, "prediction_distribution": dict(sorted(distribution.items())),
    }


def aggregate_routing(per_seed: dict[str, dict]) -> dict:
    result = {field: sample_stats([per_seed[tag][field] for tag in RUNS])
              for field in ("macro_f1", "macro_recall", "hamming_loss", "exact_match_ratio")}
    result["per_channel"] = {
        channel: {field: sample_stats([per_seed[tag]["per_channel"][channel][field] for tag in RUNS])
                  for field in ("precision", "recall", "f1", "tp", "fp", "fn", "tn")}
        for channel in CHANNELS
    }
    labels = sorted({label for value in per_seed.values() for label in value["prediction_distribution"]})
    result["prediction_distribution"] = {
        label: sample_stats([per_seed[tag]["prediction_distribution"].get(label, 0) for tag in RUNS])
        for label in labels
    }
    return result


def check_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def fixed_bag(summary: dict) -> dict:
    evaluations, source_files = {}, []
    for tag in RUNS:
        provenance = summary["individual_runs"][tag]["provenance"]
        path = Path(provenance["evaluation_json"])
        check_hash(path, provenance["evaluation_json_sha256"])
        evaluations[tag] = load(path)
        if len(evaluations[tag]["per_video"]) != 290:
            raise ValueError(f"Fixed-bag population mismatch for {tag}")
        source_files.append({"path": str(path), "sha256": provenance["evaluation_json_sha256"]})

    derived, curves, compact = {}, {}, {}
    for tag, evaluation in evaluations.items():
        rows = sorted(evaluation["per_video"], key=lambda row: row["video_id"])
        for row in rows:
            row["gated"] = row["dispatch_pred"] if row["binary_pred"] else [0, 0, 0]
            row["all_services"] = [1, 1, 1]
            row["gated_all_services"] = [1, 1, 1] if row["binary_pred"] else [0, 0, 0]
        anomalies = [row for row in rows if row["binary_true"]]
        derived[tag] = {
            "raw_all_290": routing_metrics(rows, "dispatch_pred"),
            "conditional_anomaly_140": routing_metrics(anomalies, "dispatch_pred"),
            "binary_gated_all_290": routing_metrics(rows, "gated"),
            "all_services_all_290": routing_metrics(rows, "all_services"),
            "all_services_conditional_140": routing_metrics(anomalies, "all_services"),
            "binary_gated_all_services_all_290": routing_metrics(rows, "gated_all_services"),
            "binary_gated_all_services_conditional_140": routing_metrics(anomalies, "gated_all_services"),
        }
        curves[tag] = {channel: pr_curve(
            [row["dispatch_target"][index] for row in rows],
            [row["dispatch_scores"][index] for row in rows])
            for index, channel in enumerate(CHANNELS)}
        compact[tag] = [{key: row[key] for key in (
            "video_id", "binary_true", "binary_score", "binary_pred", "dispatch_target",
            "dispatch_scores", "dispatch_pred", "gated")} for row in rows]

    conditions = {name: {
        "per_seed": {tag: derived[tag][name] for tag in RUNS},
        "aggregate": aggregate_routing({tag: derived[tag][name] for tag in RUNS}),
    } for name in derived[RUNS[0]]}
    for tag in RUNS:
        original = summary["individual_runs"][tag]
        reproduced = conditions["raw_all_290"]["per_seed"][tag]
        for key in ("macro_f1", "macro_recall", "hamming_loss", "exact_match_ratio"):
            if abs(reproduced[key] - original["metrics"][key]) > 1e-15:
                raise ValueError(f"Fixed-bag raw reproduction mismatch: {tag}/{key}")

    indexed = {tag: {row["video_id"]: row for row in evaluations[tag]["per_video"]} for tag in RUNS}
    videos = sorted(indexed[RUNS[0]])
    if any(sorted(indexed[tag]) != videos for tag in RUNS):
        raise ValueError("Fixed-bag video identities differ across seeds")
    binary_agreement, vector_agreement = [], []
    channel_agreement = {channel: [] for channel in CHANNELS}
    binary_sd, dispatch_sd = [], {channel: [] for channel in CHANNELS}
    for video in videos:
        rows = [indexed[tag][video] for tag in RUNS]
        binary_agreement.append(len({r["binary_pred"] for r in rows}) == 1)
        vector_agreement.append(len({tuple(r["dispatch_pred"]) for r in rows}) == 1)
        binary_sd.append(statistics.stdev(r["binary_score"] for r in rows))
        for index, channel in enumerate(CHANNELS):
            channel_agreement[channel].append(len({r["dispatch_pred"][index] for r in rows}) == 1)
            dispatch_sd[channel].append(statistics.stdev(r["dispatch_scores"][index] for r in rows))
    def dispersion(values: list[float]) -> dict:
        return {"mean_per_video_sample_sd": statistics.mean(values),
                "median_per_video_sample_sd": statistics.median(values),
                "max_per_video_sample_sd": max(values)}
    stability = {
        "videos": 290, "replicates": 3,
        "binary_prediction_unanimous_fraction": statistics.mean(binary_agreement),
        "dispatch_exact_vector_unanimous_fraction": statistics.mean(vector_agreement),
        "dispatch_channel_unanimous_fraction": {c: statistics.mean(channel_agreement[c]) for c in CHANNELS},
        "binary_score_dispersion": dispersion(binary_sd),
        "dispatch_score_dispersion": {c: dispersion(dispatch_sd[c]) for c in CHANNELS},
    }
    return {"scope": "test; fixed 16-instance bags; 290 videos; three Stage-2 seeds; one frozen feature store",
            "source_files": source_files, "official_raw_summary": summary,
            "derived_matched_conditions": conditions, "prediction_stability": stability,
            "per_channel_precision_recall_curves": curves, "per_video_records": compact}


def historical(summary: dict) -> dict:
    retained = {}
    for experiment_id, experiment in summary["experiments"].items():
        if experiment["classification"] != "VERIFIED_HISTORICAL_OUTPUT":
            continue
        confusion_path_text = next(path for path in experiment["files"] if path.endswith("phase2_confusion_test.json"))
        check_hash(Path(confusion_path_text), experiment["files"][confusion_path_text])
        retained[experiment_id] = {
            "classification": experiment["classification"],
            "qualification": "verified historical output; non-controlled comparison; not an official final experiment",
            "feature_source": experiment["feature_source"], "config": experiment["config"],
            "selected_epoch": experiment["selected_epoch"], "external_validation": experiment["external_val"],
            "external_test": experiment["external_test"], "test_confusion": load(Path(confusion_path_text)),
            "source_files": [{"path": path, "sha256": digest} for path, digest in experiment["files"].items()],
        }
    if set(retained) != {"hier_cascaded_v3", "hier_lora_e2e"}:
        raise ValueError("Verified historical outputs missing")
    return {"scope": summary["scope"], "task": summary["task"],
            "implementation_qualification": summary["implementation_snapshot"]["qualification"],
            "comparability": summary["comparability"], "experiments": retained}


def ablation(summary: dict) -> dict:
    return {"scope": "validation-only mechanism evidence; 161 fixed validation videos; no test evaluation",
            "evidence_boundary": summary["evidence_boundary"], "seeds": summary["seeds"],
            "standard_deviation": summary["standard_deviation"], "fixed_inputs": summary["fixed_inputs"],
            "runs": summary["runs"], "aggregates": summary["aggregates"],
            "paired_contrasts": summary["paired_contrasts"]}


def temporal(summary: dict) -> dict:
    records = {tag: {} for tag in RUNS}
    record_hashes = {entry["path"]: entry["sha256"] for entry in summary["record_files"]}
    example_source = None
    for path_text, expected in record_hashes.items():
        path = Path(path_text)
        check_hash(path, expected)
        record = load(path)
        records[record["checkpoint"]][record["video_id"]] = record
    if any(len(records[tag]) != 290 for tag in RUNS):
        raise ValueError("Phase-5 per-video record population mismatch")
    binary_scores, aurocs = {}, {}
    for tag in RUNS:
        ordered = [records[tag][video] for video in sorted(records[tag])]
        labels = [r["routing"]["binary_true"] for r in ordered]
        scores = [r["routing"]["binary_score"] for r in ordered]
        actual, expected = _roc_auc(labels, scores), summary["checkpoint_summaries"][tag]["binary"]["auroc"]
        if abs(actual - expected) > 1e-15:
            raise ValueError(f"Exact Phase-5 AUROC mismatch for {tag}: {actual} != {expected}")
        binary_scores[tag] = {"video_id": [r["video_id"] for r in ordered],
                              "binary_true": labels, "binary_score": scores}
        aurocs[tag] = {"recomputed": actual, "canonical": expected,
                       "absolute_difference": abs(actual - expected), "verified_within_1e-15": True}
    example = next(record for video, record in sorted(records[RUNS[0]].items())
                   if record["annotation_status"] == "anomaly" and record["routing"]["alert_triggered"])
    for path_text, expected in record_hashes.items():
        record = load(Path(path_text))
        if record["checkpoint"] == RUNS[0] and record["video_id"] == example["video_id"]:
            example_source = {"path": path_text, "sha256": expected}
            break
    qualitative = {"selection_rule": "lexicographically first eligible anomalous s1337 record with an alert; not performance-selected",
                   "label": "offline demonstration, not deployment", "video_id": example["video_id"],
                   "metadata": example["metadata"], "intervals": example["intervals"],
                   "temporal_metrics": example["temporal_metrics"], "coverage": example["coverage"],
                   "routing": example["routing"], "record_provenance": example_source}
    return {"scope": "dense indexed-system test evaluation; three fixed official checkpoints",
            "population": summary["population"], "protocol_sha256": summary["protocol_sha256"],
            "attempt_id": summary["attempt_id"], "feature_store_identity": summary["feature_store_identity"],
            "teacher_sha256": summary["teacher_sha256"], "failures": summary["failures"],
            "checkpoint_summaries": summary["checkpoint_summaries"],
            "three_checkpoint_statistics": summary["three_checkpoint_statistics"],
            "temporal_baselines": summary["temporal_baselines"], "coverage": summary["coverage"],
            "category_coverage": summary["category_coverage"], "record_files": summary["record_files"],
            "binary_scores": binary_scores, "auroc_verification": aurocs,
            "qualitative_cad_example": qualitative}


TAXONOMY = {
    "fixed_bag_raw": "Fixed 16-clip bags; raw dispatch; all 290 test videos.",
    "fixed_bag_conditional": "Fixed 16-clip bags; raw dispatch; 140 anomalous test videos only.",
    "fixed_bag_binary_gated": "Fixed 16-clip bags; dispatch zeroed after a negative binary decision; all 290 videos.",
    "indexed_system_raw": "Dense variable-length indexed bags; raw primary dispatch; all 290 videos.",
    "indexed_system_conditional": "Dense indexed bags; raw primary dispatch; 140 anomalous videos only.",
    "indexed_system_binary_gated": "Dense indexed bags; primary dispatch gated by binary decision; all 290 videos.",
    "all_services_baseline": "All services predicted; population stated with each result.",
    "binary_gated_all_services_baseline": "All services after positive binary decision, otherwise none; population stated.",
    "temporal_grounding": "135 source-bounded anomalies; five conflicts excluded only from temporal metrics.",
    "validation_teacher_ablation": "Validation-only mechanism evidence; not test/generalization evidence.",
    "historical_14_class": "Verified historical single-label output; non-controlled and not directly comparable.",
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite canonical package: {OUTPUT}")
    paths = {"phase1_dispatch_summary": DISPATCH, "phase2_historical_summary": HISTORICAL,
             "phase3_ablation_summary": ABLATION, "phase5_temporal_summary": TEMPORAL}
    sources = {name: load(path) for name, path in paths.items()}
    package = {
        "schema_version": 1, "package_id": "canonical_thesis_v1", "created_on": "2026-09-19",
        "authority": "sole canonical quantitative source for final thesis writing after Phase 6",
        "source_files": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "metric_taxonomy": TAXONOMY,
        "phase1_fixed_bag_dispatch": fixed_bag(sources["phase1_dispatch_summary"]),
        "phase2_historical_14_class": historical(sources["phase2_historical_summary"]),
        "phase3_teacher_ablation": ablation(sources["phase3_ablation_summary"]),
        "phase5_temporal_system": temporal(sources["phase5_temporal_summary"]),
        "figure_context": {
            "pipeline_stages": ["Indexed video", "Deterministic clips", "LoRA-adapted frozen VideoMAE features",
                                "Normal-prototype distance", "Attention MIL binary + response heads",
                                "Binary gate", "Offline CAD-style alert"],
            "protocol_flows": {"train": "fixed inputs -> three Stage-2 seeds",
                               "validation": "checkpoint selection + validation-only teacher ablation",
                               "test": "fixed bags + frozen dense protocol -> one evaluation per checkpoint"}},
        "binding_limitations": [
            "Temporal annotations use the declared frozen coordinate convention; five source-boundary conflicts remain uncorrected.",
            "Police/Medical/Fire targets are project-derived, not observed service ground truth.",
            "Transformers 5.12.0 retains verified zero-new-bias compatibility behavior.",
            "Persisted float16 prototypes approximate unavailable original float32 centers.",
            "n=3 supports descriptive mean and sample SD, not inferential significance claims.",
            "The test set was previously inspected and is not an untouched selection resource.",
            "Large dense artifacts remain local/Git-ignored without demonstrated external restore.",
            "CAD-style behavior is an offline demonstration, not deployment evidence.",
            "Phase-3 teacher ablation is validation-only mechanism evidence."]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(package, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"PASS: wrote {OUTPUT} SHA-256 {sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
