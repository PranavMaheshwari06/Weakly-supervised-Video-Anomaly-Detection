#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import statistics
import sys
from pathlib import Path

import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2_class.src.utils.provenance import sha256_file, sha256_json


OUTPUT = ROOT / "artifacts/stage2/evidence/teacher_ablation_v1/summary.json"
SEEDS = (1337, 2027, 3407)
VARIANTS = {
    "full": {
        "config": "stage2_class/configs/teacher_ablation_full.yaml",
        "inject_teacher_score": True,
        "mixture_uniform_ratio": 0.5,
        "top_fraction": 0.25,
    },
    "no_input": {
        "config": "stage2_class/configs/teacher_ablation_no_input.yaml",
        "inject_teacher_score": False,
        "mixture_uniform_ratio": 0.5,
        "top_fraction": 0.25,
    },
    "uniform": {
        "config": "stage2_class/configs/teacher_ablation_uniform.yaml",
        "inject_teacher_score": True,
        "mixture_uniform_ratio": 1.0,
        "top_fraction": 0.0,
    },
    "neither": {
        "config": "stage2_class/configs/teacher_ablation_neither.yaml",
        "inject_teacher_score": False,
        "mixture_uniform_ratio": 1.0,
        "top_fraction": 0.0,
    },
}
METRICS = ("macro_f1", "binary_auroc", "binary_f1", "hamming_loss")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def recompute_metrics(per_video: list[dict]) -> dict[str, float]:
    expect(len(per_video) == 161, "Every validation evaluation must contain 161 videos")
    targets = [row["dispatch_target"] for row in per_video]
    preds = [row["dispatch_pred"] for row in per_video]
    f1s = []
    for channel in range(3):
        tp = sum(t[channel] == 1 and p[channel] == 1 for t, p in zip(targets, preds))
        fp = sum(t[channel] == 0 and p[channel] == 1 for t, p in zip(targets, preds))
        fn = sum(t[channel] == 1 and p[channel] == 0 for t, p in zip(targets, preds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    binary_targets = [int(row["binary_true"]) for row in per_video]
    binary_preds = [int(row["binary_pred"]) for row in per_video]
    binary_scores = [float(row["binary_score"]) for row in per_video]
    tp = sum(t == 1 and p == 1 for t, p in zip(binary_targets, binary_preds))
    fp = sum(t == 0 and p == 1 for t, p in zip(binary_targets, binary_preds))
    fn = sum(t == 1 and p == 0 for t, p in zip(binary_targets, binary_preds))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    binary_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = sum(
        target != pred
        for target_row, pred_row in zip(targets, preds)
        for target, pred in zip(target_row, pred_row)
    )
    return {
        "macro_f1": sum(f1s) / 3,
        "binary_auroc": float(roc_auc_score(binary_targets, binary_scores)),
        "binary_f1": binary_f1,
        "hamming_loss": errors / (len(per_video) * 3),
    }


def config_without_factors(config: dict) -> dict:
    normalized = copy.deepcopy(config)
    for key in ("inject_teacher_score", "mixture_uniform_ratio", "top_fraction"):
        normalized["stage2"].pop(key)
    return normalized


def aggregate(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite canonical evidence: {OUTPUT}")

    configs = {}
    stripped_identities = set()
    for variant, specification in VARIANTS.items():
        path = ROOT / specification["config"]
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in ("inject_teacher_score", "mixture_uniform_ratio", "top_fraction"):
            expect(config["stage2"][key] == specification[key], f"{variant}: factor mismatch for {key}")
        expect(config["reproducibility"].get("validation_only") is True, f"{variant}: validation-only guard")
        stripped_identities.add(sha256_json(config_without_factors(config)))
        configs[variant] = {
            "path": specification["config"],
            "sha256": sha256_file(path),
            "factors": {key: specification[key] for key in (
                "inject_teacher_score", "mixture_uniform_ratio", "top_fraction"
            )},
        }
    expect(len(stripped_identities) == 1, "Normalized source configs differ outside declared factors")

    runs = []
    plans: dict[tuple[str, int], dict] = {}
    for variant, specification in VARIANTS.items():
        for seed in SEEDS:
            run_tag = f"ablation_{variant}_s{seed}"
            run_root = ROOT / "artifacts/stage2/runs" / run_tag
            eval_root = ROOT / "artifacts/stage2/eval" / run_tag
            expected_eval = eval_root / "dispatch_eval_val.json"
            expect(sorted(path.name for path in eval_root.glob("*.json")) == [expected_eval.name],
                   f"{run_tag}: expected exactly one validation JSON")

            manifest_path = run_root / "run_manifest.json"
            resolved_path = run_root / "resolved_config.yaml"
            plan_path = run_root / "training_bag_plan.json"
            log_path = run_root / "train/dispatch_train_log.jsonl"
            train_summary_path = run_root / "train/dispatch_summary.json"
            checkpoint_path = run_root / "checkpoints/mil_best.pt"
            manifest = load_json(manifest_path)
            train_summary = load_json(train_summary_path)
            evaluation = load_json(expected_eval)
            plan = load_json(plan_path)
            resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))

            expect(manifest["run_tag"] == run_tag and manifest["training_seed"] == seed,
                   f"{run_tag}: manifest identity")
            expect(manifest.get("validation_only") is True, f"{run_tag}: validation-only manifest")
            expect(set(manifest["input_paths"]["teacher_scores"]) == {"train", "val"},
                   f"{run_tag}: test teacher path present")
            expect(set(manifest["input_paths"]["evaluation_bag_manifests"]) == {"val"},
                   f"{run_tag}: test manifest path present")
            expect(set(manifest["artifact_hashes"]["feature_store"]) == {"train", "val"},
                   f"{run_tag}: test feature identity present")
            expect(not any("test" in path.lower() for path in manifest["artifact_hashes"]["files"]),
                   f"{run_tag}: test file hash present")
            expect(manifest["bag_settings"]["test"] == "embargoed_not_loaded", f"{run_tag}: embargo")
            expect(manifest["threshold_policy"]["dispatch_primary"] == 0.5, f"{run_tag}: threshold")
            expect(manifest["early_stopping"]["patience"] == 15, f"{run_tag}: patience")
            expect(manifest["early_stopping"]["minimum_delta"] == 1e-4, f"{run_tag}: min delta")
            expect(manifest["early_stopping"]["maximum_epochs"] == 100, f"{run_tag}: max epochs")
            expect(resolved["seed"] == seed, f"{run_tag}: resolved seed")
            expect(sha256_file(resolved_path) == manifest["configuration_identity"],
                   f"{run_tag}: resolved config identity")
            expect(train_summary["selected_epoch"] == manifest["early_stopping"]["selected_epoch"],
                   f"{run_tag}: selected epoch")

            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            expect(len(records) == manifest["epochs_ran"], f"{run_tag}: epoch-log count")
            selected = records[train_summary["selected_epoch"]]
            expect(abs(selected["val_macro_f1"] - train_summary["best_val_macro_f1"]) < 1e-12,
                   f"{run_tag}: selected metric")

            metrics = recompute_metrics(evaluation["per_video"])
            for metric, value in metrics.items():
                expect(abs(value - float(evaluation[metric])) < 1e-12,
                       f"{run_tag}: saved {metric} does not recompute")
            expect(abs(metrics["macro_f1"] - train_summary["best_val_macro_f1"]) < 1e-12,
                   f"{run_tag}: selected checkpoint validation mismatch")

            expect(plan["metadata"]["training_seed"] == seed, f"{run_tag}: bag-plan seed")
            expect(plan["metadata"]["maximum_epochs"] == 100, f"{run_tag}: bag-plan epochs")
            expect(plan["metadata"]["bags_per_epoch"] == 1449, f"{run_tag}: bags per epoch")
            expect(plan["metadata"]["planned_bag_count"] == 144900, f"{run_tag}: planned bags")
            expect(plan["plan_identity_sha256"] == sha256_json({
                "schema_version": plan["schema_version"],
                "metadata": plan["metadata"],
                "epoch_identities": plan["epoch_identities"],
            }), f"{run_tag}: bag-plan identity")
            plans[(variant, seed)] = plan

            artifact_paths = [manifest_path, resolved_path, plan_path, log_path,
                              train_summary_path, checkpoint_path, expected_eval]
            runs.append({
                "variant": variant,
                "seed": seed,
                "run_tag": run_tag,
                "selected_epoch": train_summary["selected_epoch"],
                "epochs_ran": train_summary["epochs_ran"],
                "metrics": metrics,
                "training_command": manifest["execution"]["command"],
                "evaluation_command": (
                    "venv/bin/python -m stage2_class.src.eval.m29_eval_dispatch "
                    f"--split val --run_tag {run_tag}"
                ),
                "training_bag_plan": manifest["training_bag_plan"],
                "artifact_hashes": {
                    path.relative_to(ROOT).as_posix(): sha256_file(path) for path in artifact_paths
                },
            })

    paired_plan_hashes = {}
    common_plan_keys = (
        "training_seed", "sampling_context", "bag_size", "warmup_epochs",
        "maximum_epochs", "eligible_video_count", "bags_per_epoch", "planned_bag_count",
        "video_ids_sha256", "source_feature_identity_sha256", "teacher_scores_sha256",
    )
    for seed in SEEDS:
        full = plans[("full", seed)]
        no_input = plans[("no_input", seed)]
        uniform = plans[("uniform", seed)]
        neither = plans[("neither", seed)]
        expect(full == no_input, f"seed {seed}: guided pair bag plans differ")
        expect(uniform == neither, f"seed {seed}: uniform pair bag plans differ")
        for key in common_plan_keys:
            expect(full["metadata"][key] == uniform["metadata"][key],
                   f"seed {seed}: guided/uniform non-factor mismatch: {key}")
        paired_plan_hashes[str(seed)] = {
            "guided_file_sha256": runs[[r["run_tag"] for r in runs].index(f"ablation_full_s{seed}")]
                ["training_bag_plan"]["file_sha256"],
            "uniform_file_sha256": runs[[r["run_tag"] for r in runs].index(f"ablation_uniform_s{seed}")]
                ["training_bag_plan"]["file_sha256"],
            "eligible_videos": full["metadata"]["eligible_video_count"],
            "bags_per_epoch": full["metadata"]["bags_per_epoch"],
            "maximum_epochs": full["metadata"]["maximum_epochs"],
        }

    aggregates = {}
    for variant in VARIANTS:
        variant_runs = [run for run in runs if run["variant"] == variant]
        aggregates[variant] = {
            metric: aggregate([run["metrics"][metric] for run in variant_runs])
            for metric in METRICS
        }

    contrasts = {}
    for name, left, right in (
        ("score_input_effect_under_guided", "full", "no_input"),
        ("score_input_effect_under_uniform", "uniform", "neither"),
        ("guided_sampling_effect_with_input", "full", "uniform"),
        ("guided_sampling_effect_without_input", "no_input", "neither"),
    ):
        contrasts[name] = {}
        for metric in METRICS:
            differences = []
            by_seed = {}
            for seed in SEEDS:
                left_value = next(r["metrics"][metric] for r in runs if r["variant"] == left and r["seed"] == seed)
                right_value = next(r["metrics"][metric] for r in runs if r["variant"] == right and r["seed"] == seed)
                by_seed[str(seed)] = left_value - right_value
                differences.append(left_value - right_value)
            contrasts[name][metric] = {"left_minus_right_by_seed": by_seed, **aggregate(differences)}

    protected_test_hashes = {
        "artifacts/stage2/teacher/teacher_scores_test.pt": sha256_file(
            ROOT / "artifacts/stage2/teacher/teacher_scores_test.pt"
        ),
        "data/ucf_crime/splits/dman_v21/test.txt": sha256_file(
            ROOT / "data/ucf_crime/splits/dman_v21/test.txt"
        ),
    }
    payload = {
        "schema_version": 1,
        "evidence_id": "teacher_ablation_v1",
        "evidence_boundary": "Validation-only mechanism evidence; no test evaluation.",
        "seeds": list(SEEDS),
        "metrics": list(METRICS),
        "standard_deviation": "sample (n-1)",
        "configs": configs,
        "common_config_identity_excluding_factors": next(iter(stripped_identities)),
        "fixed_inputs": {
            "train_split_sha256": sha256_file(ROOT / "data/ucf_crime/splits/dman_v21/train.txt"),
            "validation_split_sha256": sha256_file(ROOT / "data/ucf_crime/splits/dman_v21/val.txt"),
            "validation_bag_manifest_sha256": sha256_file(
                ROOT / "artifacts/stage2/bag_manifests/dispatch_lora_fixed_val.json"
            ),
            "teacher_train_sha256": sha256_file(ROOT / "artifacts/stage2/teacher/teacher_scores_train.pt"),
            "teacher_validation_sha256": sha256_file(ROOT / "artifacts/stage2/teacher/teacher_scores_val.pt"),
            "feature_store_identity_sha256": {
                "train": "7a4fb455d9fe7741fee7bb99fb5fea437de7f4e30beca8912e6971c56bf31788",
                "val": "c0a8ed81ccfe39ab86468a2d20018ca13d642dd4ac77ec1fc70cc44ebb020c41",
            },
        },
        "protected_test_preflight_hashes_not_loaded_by_runs": protected_test_hashes,
        "implementation_hashes": {
            path: sha256_file(ROOT / path) for path in (
                "stage2_class/src/train/m27_train_dispatch_mil.py",
                "stage2_class/src/eval/m29_eval_dispatch.py",
                "stage2_class/src/data/mg_mil_sampler.py",
                "stage2_class/src/io/training_bag_plan.py",
                "stage2_class/src/train/engine.py",
                "stage2_class/src/models/mil_hierarchical.py",
            )
        },
        "paired_training_bag_plans": paired_plan_hashes,
        "runs": runs,
        "aggregates": aggregates,
        "paired_contrasts": contrasts,
        "failed_runs": [],
        "reruns": [],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
