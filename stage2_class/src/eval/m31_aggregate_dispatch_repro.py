from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


RUN_TAGS = (
    "dispatch_repro_s1337",
    "dispatch_repro_s2027",
    "dispatch_repro_s3407",
)
TEST_MANIFEST = Path("artifacts/stage2/bag_manifests/dispatch_lora_fixed_test.json")
TEST_MANIFEST_SHA256 = "9cddad6d40e9efe8f11c51982199ea1a2056a1c5be6131b4d4df083a37fc4041"
OUTPUT_PATH = Path("artifacts/stage2/evidence/dispatch_repro_v1/summary.json")

METRIC_NAMES = (
    "macro_f1",
    "macro_recall",
    "hamming_loss",
    "exact_match_ratio",
    "binary_auroc",
    "binary_f1",
    "binary_accuracy",
    "binary_precision",
    "binary_recall",
)
CHANNEL_FIELDS = (
    "precision",
    "recall",
    "f1",
    "support_pos",
    "support_neg",
    "tp",
    "fp",
    "fn",
    "tn",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_statistics(values: list[float | int]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def load_run(run_tag: str) -> dict:
    eval_path = Path("artifacts/stage2/eval") / run_tag / "dispatch_eval_test.json"
    run_root = Path("artifacts/stage2/runs") / run_tag
    checkpoint_path = run_root / "checkpoints/mil_best.pt"

    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    run_manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    provenance = evaluation["provenance"]

    if provenance["run_tag"] != run_tag:
        raise ValueError(f"Evaluation run tag mismatch for {run_tag}")
    if provenance["bag_manifest"] != str(TEST_MANIFEST):
        raise ValueError(f"Unexpected test manifest for {run_tag}")
    if provenance["bag_manifest_sha256"] != TEST_MANIFEST_SHA256:
        raise ValueError(f"Test manifest hash mismatch for {run_tag}")
    if provenance["threshold_policy"]["dispatch_primary"] != 0.5:
        raise ValueError(f"Dispatch threshold mismatch for {run_tag}")
    if provenance["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError(f"Checkpoint hash mismatch for {run_tag}")

    return {
        "training_seed": run_manifest["training_seed"],
        "count_videos": evaluation["count_videos"],
        "metrics": {name: evaluation[name] for name in METRIC_NAMES},
        "binary_confusion": evaluation["binary_confusion"],
        "per_channel": evaluation["per_channel"],
        "dispatch_prediction_distribution": evaluation["dispatch_prediction_distribution"],
        "binary_prediction_distribution": evaluation["binary_prediction_distribution"],
        "provenance": {
            **provenance,
            "evaluation_json": str(eval_path),
            "evaluation_json_sha256": sha256_file(eval_path),
        },
    }


def aggregate_metrics(runs: dict[str, dict]) -> dict:
    return {
        name: sample_statistics([run["metrics"][name] for run in runs.values()])
        for name in METRIC_NAMES
    }


def aggregate_channels(runs: dict[str, dict]) -> dict:
    aggregate = {}
    for channel in ("police", "medical", "fire"):
        aggregate[channel] = {
            field: sample_statistics(
                [run["per_channel"][channel][field] for run in runs.values()]
            )
            for field in CHANNEL_FIELDS
        }
    return aggregate


def aggregate_distribution(runs: dict[str, dict], distribution_name: str) -> dict:
    labels = sorted(
        {
            label
            for run in runs.values()
            for label in run[distribution_name]
        }
    )
    aggregate = {}
    for label in labels:
        counts = [run[distribution_name].get(label, 0) for run in runs.values()]
        aggregate[label] = {
            "combined_count": sum(counts),
            "mean_count": statistics.mean(counts),
            "sample_std_count": statistics.stdev(counts),
        }
    return aggregate


def main() -> None:
    actual_manifest_hash = sha256_file(TEST_MANIFEST)
    if actual_manifest_hash != TEST_MANIFEST_SHA256:
        raise ValueError("Fixed test manifest hash does not match the frozen protocol")
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite evidence summary: {OUTPUT_PATH}")

    runs = {run_tag: load_run(run_tag) for run_tag in RUN_TAGS}
    summary = {
        "schema_version": 1,
        "evaluation": {
            "split": "test",
            "num_seeds": len(runs),
            "run_tags": list(RUN_TAGS),
            "dispatch_threshold": 0.5,
            "fixed_bag_manifest": str(TEST_MANIFEST),
            "fixed_bag_manifest_sha256": actual_manifest_hash,
            "standard_deviation": "sample (N-1)",
        },
        "individual_runs": runs,
        "aggregate": {
            "metrics": aggregate_metrics(runs),
            "per_channel": aggregate_channels(runs),
            "prediction_distributions": {
                name: aggregate_distribution(runs, name)
                for name in (
                    "dispatch_prediction_distribution",
                    "binary_prediction_distribution",
                )
            },
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
