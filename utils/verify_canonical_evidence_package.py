"""Independently verify the Phase-6 canonical package and generated figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, multilabel_confusion_matrix, precision_recall_fscore_support, roc_auc_score

PACKAGE = Path("artifacts/stage2/evidence/canonical_thesis_v1/package.json")
FIGURES = Path("artifacts/thesis_figures/phase6")
RUNS = ("dispatch_repro_s1337", "dispatch_repro_s2027", "dispatch_repro_s3407")
CHANNELS = ("police", "medical", "fire")
REQUIRED_FIGURES = (
    "fig_architecture_dataflow.pdf", "fig_protocol_provenance.pdf", "fig_headline_metrics.pdf",
    "fig_binary_roc.pdf", "fig_per_channel_pr_curves.pdf", "fig_per_channel_prf1.pdf",
    "fig_joint_dispatch_distribution.pdf", "fig_routing_comparison.pdf",
    "fig_teacher_ablation.pdf", "fig_historical_14class.pdf",
    "fig_temporal_ap_by_category.pdf", "fig_temporal_iou_thresholds.pdf",
    "fig_temporal_baselines_comparison.pdf", "fig_coverage_comparison.pdf",
    "fig_temporal_cad_example.pdf", "fig_prediction_stability.pdf",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, name: str, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{name}: {actual} != {expected}")


def routing(rows: list[dict], field: str) -> dict:
    truth = np.asarray([row["dispatch_target"] for row in rows], dtype=int)
    pred = np.asarray([row[field] for row in rows], dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, average=None, zero_division=0)
    confusion = multilabel_confusion_matrix(truth, pred)
    return {"macro_f1": float(f1.mean()), "macro_recall": float(recall.mean()),
            "hamming_loss": float((truth != pred).mean()),
            "exact_match_ratio": float(np.all(truth == pred, axis=1).mean()),
            "per_channel": {channel: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "tn": int(confusion[i, 0, 0]), "fp": int(confusion[i, 0, 1]),
                "fn": int(confusion[i, 1, 0]), "tp": int(confusion[i, 1, 1])}
                for i, channel in enumerate(CHANNELS)}}


def verify_fixed(package: dict) -> None:
    fixed = package["phase1_fixed_bag_dispatch"]
    source_summary = load(Path(package["source_files"]["phase1_dispatch_summary"]["path"]))
    evaluations = {}
    for tag in RUNS:
        path = Path(source_summary["individual_runs"][tag]["provenance"]["evaluation_json"])
        evaluations[tag] = load(path)
        rows = evaluations[tag]["per_video"]
        for row in rows:
            row["gated"] = row["dispatch_pred"] if row["binary_pred"] else [0, 0, 0]
            row["all_services"] = [1, 1, 1]
            row["gated_all_services"] = [1, 1, 1] if row["binary_pred"] else [0, 0, 0]
        anomaly = [row for row in rows if row["binary_true"]]
        definitions = {
            "raw_all_290": (rows, "dispatch_pred"), "conditional_anomaly_140": (anomaly, "dispatch_pred"),
            "binary_gated_all_290": (rows, "gated"), "all_services_all_290": (rows, "all_services"),
            "all_services_conditional_140": (anomaly, "all_services"),
            "binary_gated_all_services_all_290": (rows, "gated_all_services"),
            "binary_gated_all_services_conditional_140": (anomaly, "gated_all_services")}
        for condition, (population, field) in definitions.items():
            actual = routing(population, field)
            saved = fixed["derived_matched_conditions"][condition]["per_seed"][tag]
            for metric in ("macro_f1", "macro_recall", "hamming_loss", "exact_match_ratio"):
                close(actual[metric], saved[metric], f"{tag}/{condition}/{metric}")
            for channel in CHANNELS:
                for metric in ("precision", "recall", "f1", "tp", "fp", "fn", "tn"):
                    close(actual["per_channel"][channel][metric], saved["per_channel"][channel][metric],
                          f"{tag}/{condition}/{channel}/{metric}")
        for index, channel in enumerate(CHANNELS):
            labels = [row["dispatch_target"][index] for row in rows]
            scores = [row["dispatch_scores"][index] for row in rows]
            curve = fixed["per_channel_precision_recall_curves"][tag][channel]
            close(float(average_precision_score(labels, scores)),
                  curve["average_precision"],
                  f"{tag}/{channel}/average_precision")
            assert curve["precision"][0] == 1.0 and curve["recall"][0] == 0.0
            for point, threshold in enumerate(curve["thresholds"], start=1):
                prediction = [score >= threshold for score in scores]
                tp = sum(truth and pred for truth, pred in zip(labels, prediction))
                fp = sum(not truth and pred for truth, pred in zip(labels, prediction))
                close(tp / (tp + fp), curve["precision"][point], f"{tag}/{channel}/PR precision")
                close(tp / sum(labels), curve["recall"][point], f"{tag}/{channel}/PR recall")

    for condition, result in fixed["derived_matched_conditions"].items():
        for metric in ("macro_f1", "macro_recall", "hamming_loss", "exact_match_ratio"):
            values = [result["per_seed"][tag][metric] for tag in RUNS]
            aggregate = result["aggregate"][metric]
            close(statistics.mean(values), aggregate["mean"], f"{condition}/{metric}/mean")
            close(statistics.stdev(values), aggregate["sample_sd"], f"{condition}/{metric}/sd")

    videos = sorted(row["video_id"] for row in evaluations[RUNS[0]]["per_video"])
    by_tag = {tag: {row["video_id"]: row for row in evaluations[tag]["per_video"]} for tag in RUNS}
    stability = fixed["prediction_stability"]
    binary = [len({by_tag[tag][video]["binary_pred"] for tag in RUNS}) == 1 for video in videos]
    exact = [len({tuple(by_tag[tag][video]["dispatch_pred"]) for tag in RUNS}) == 1 for video in videos]
    close(statistics.mean(binary), stability["binary_prediction_unanimous_fraction"], "binary stability")
    close(statistics.mean(exact), stability["dispatch_exact_vector_unanimous_fraction"], "vector stability")
    binary_sd = [statistics.stdev(by_tag[tag][video]["binary_score"] for tag in RUNS) for video in videos]
    close(statistics.mean(binary_sd), stability["binary_score_dispersion"]["mean_per_video_sample_sd"], "binary score dispersion mean")
    close(statistics.median(binary_sd), stability["binary_score_dispersion"]["median_per_video_sample_sd"], "binary score dispersion median")
    close(max(binary_sd), stability["binary_score_dispersion"]["max_per_video_sample_sd"], "binary score dispersion max")
    for index, channel in enumerate(CHANNELS):
        values = [len({by_tag[tag][video]["dispatch_pred"][index] for tag in RUNS}) == 1 for video in videos]
        close(statistics.mean(values), stability["dispatch_channel_unanimous_fraction"][channel], f"{channel} stability")
        score_sd = [statistics.stdev(by_tag[tag][video]["dispatch_scores"][index] for tag in RUNS) for video in videos]
        saved = stability["dispatch_score_dispersion"][channel]
        close(statistics.mean(score_sd), saved["mean_per_video_sample_sd"], f"{channel} score dispersion mean")
        close(statistics.median(score_sd), saved["median_per_video_sample_sd"], f"{channel} score dispersion median")
        close(max(score_sd), saved["max_per_video_sample_sd"], f"{channel} score dispersion max")


def verify_historical(package: dict) -> None:
    section = package["phase2_historical_14_class"]
    assert set(section["experiments"]) == {"hier_cascaded_v3", "hier_lora_e2e"}
    expected = {"hier_cascaded_v3": 0.07395382395382395, "hier_lora_e2e": 0.19611839475832898}
    for name, value in expected.items():
        experiment = section["experiments"][name]
        assert experiment["classification"] == "VERIFIED_HISTORICAL_OUTPUT"
        close(experiment["external_test"]["macro_f1"], value, f"{name} historical Macro-F1", 0.0)
        matrix = np.asarray(experiment["test_confusion"]["matrix"])
        assert matrix.shape == (14, 14) and int(matrix.sum()) == 290


def verify_ablation(package: dict) -> None:
    section = package["phase3_teacher_ablation"]
    assert "validation-only" in section["scope"] and len(section["runs"]) == 12
    by_variant = {name: [] for name in ("full", "no_input", "uniform", "neither")}
    for run in section["runs"]:
        by_variant[run["variant"]].append(run)
    for variant, runs in by_variant.items():
        assert len(runs) == 3
        for metric in ("macro_f1", "binary_auroc", "binary_f1", "hamming_loss"):
            values = [run["metrics"][metric] for run in runs]
            saved = section["aggregates"][variant][metric]
            close(statistics.mean(values), saved["mean"], f"{variant}/{metric}/mean")
            close(statistics.stdev(values), saved["sample_std"], f"{variant}/{metric}/sd")


def verify_temporal(package: dict) -> None:
    section = package["phase5_temporal_system"]
    assert section["population"]["videos"] == 290 and section["population"]["temporal_eligible_anomaly"] == 135
    assert section["failures"] == [] and len(section["record_files"]) == 870
    for entry in section["record_files"]:
        assert sha(Path(entry["path"])) == entry["sha256"]
    expected = (0.9402380952380952, 0.9415714285714286, 0.9429523809523809)
    for tag, canonical in zip(RUNS, expected):
        scores = section["binary_scores"][tag]
        actual = float(roc_auc_score(scores["binary_true"], scores["binary_score"]))
        close(actual, canonical, f"{tag}/AUROC", 0.0)
        close(section["auroc_verification"][tag]["canonical"], canonical, f"{tag}/canonical AUROC", 0.0)
    coverage = section["coverage"]
    for condition in ("dense_realized", "sparse", "fixed_bag"):
        item = coverage[condition]
        close(item["covered_source_frames"] / item["total_source_frames"], item["frame_weighted_coverage"],
              f"{condition}/coverage")
    for baseline in section["temporal_baselines"].values():
        assert "sample_sd" not in json.dumps(baseline)
    stats_section = section["three_checkpoint_statistics"]
    for family in stats_section.values():
        if not isinstance(family, dict):
            continue
        stack = [family]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                if set(("values", "mean", "sample_sd", "n")).issubset(value):
                    assert value["n"] == 3 and len(value["values"]) == 3
                    close(statistics.mean(value["values"]), value["mean"], "Phase-5 aggregate mean")
                    close(statistics.stdev(value["values"]), value["sample_sd"], "Phase-5 aggregate SD")
                else:
                    stack.extend(value.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-figures", action="store_true")
    args = parser.parse_args()
    package = load(PACKAGE)
    assert package["package_id"] == "canonical_thesis_v1" and package["schema_version"] == 1
    assert len(package["metric_taxonomy"]) == 11 and len(package["binding_limitations"]) == 9
    for source in package["source_files"].values():
        # The package records the mirror's location before it moved to reproducibility/evidence.
        path = Path(source["path"].replace("RP_docs/evidence", "reproducibility/evidence"))
        assert sha(path) == source["sha256"]
    verify_fixed(package)
    verify_historical(package)
    verify_ablation(package)
    verify_temporal(package)
    figure_hashes = {}
    if args.require_figures:
        for name in REQUIRED_FIGURES:
            path = FIGURES / name
            assert path.is_file() and path.stat().st_size > 1000
            assert path.read_bytes().startswith(b"%PDF-")
            figure_hashes[name] = sha(path)
    print(json.dumps({"status": "pass", "package_sha256": sha(PACKAGE), "source_summaries": 4,
                      "fixed_bag_records": 870, "temporal_records": 870,
                      "historical_outputs": 2, "ablation_runs": 12,
                      "figures": figure_hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
