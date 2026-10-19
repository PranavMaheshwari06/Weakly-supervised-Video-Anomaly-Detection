"""
Dispatch MIL Evaluation.

Evaluates the trained dispatch model on val/test split:
- Per-channel (police/medical/fire) Precision, Recall, F1
- Macro-F1, Hamming Loss, Exact Match Ratio
- Binary AUC-ROC
- Confusion per dispatch channel

Usage:
    python -m stage2_class.src.eval.m29_eval_dispatch --split val --run_tag dispatch_repro_seed1337
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score

from stage2_class.src.io.fixed_bag_manifest import (
    FixedBagSampler,
    load_fixed_bag_sampler,
    manifest_path,
)
from stage2_class.src.data.mil_dataset import MilVideoDataset
from stage2_class.src.data.mil_index import (
    build_clip_lookup_for_split,
    build_video_to_clips_index,
    load_teacher_scores,
)
from stage2_class.src.io.split_reader import read_split_txt
from stage2_class.src.io.videos_index import load_videos_index
from stage2_class.src.models.mil_hierarchical import HierarchicalAttentionMilHead
from stage2_class.src.utils.dispatch_label_map import (
    DISPATCH_CHANNELS,
    NUM_DISPATCH,
    PRIMARY_DISPATCH_THRESHOLD,
    get_dispatch_label,
)
from stage2_class.src.utils.provenance import (
    feature_store_identity,
    sha256_file,
    validate_run_tag,
)
from stage2_class.src.utils.seed import set_deterministic


def resolve_checkpoint(run_tag: str) -> Path:
    p = Path("artifacts/stage2/runs") / run_tag / "checkpoints" / "mil_best.pt"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def compute_metrics(
    targets_dispatch,
    preds_dispatch,
    targets_binary,
    scores_binary,
    preds_binary,
) -> Dict:
    """Compute dispatch and binary metrics without model or file-system state."""
    dispatch_targets = [list(map(int, row)) for row in targets_dispatch]
    dispatch_preds = [list(map(int, row)) for row in preds_dispatch]
    binary_targets = [int(value) for value in targets_binary]
    binary_scores = [float(value) for value in scores_binary]
    binary_preds = [int(value) for value in preds_binary]

    count = len(dispatch_targets)
    if count == 0:
        raise ValueError("Cannot compute metrics for an empty evaluation set")
    if not all(
        len(values) == count
        for values in (dispatch_preds, binary_targets, binary_scores, binary_preds)
    ):
        raise ValueError("Metric inputs must contain the same number of videos")
    if any(len(row) != NUM_DISPATCH for row in dispatch_targets + dispatch_preds):
        raise ValueError(f"Dispatch metric inputs must have {NUM_DISPATCH} channels")
    if set(binary_targets) != {0, 1}:
        raise ValueError("Binary AUROC requires both target classes 0 and 1")

    channel_metrics = {}
    f1s = []
    recalls = []
    for channel_index, channel_name in enumerate(DISPATCH_CHANNELS):
        target_values = [row[channel_index] for row in dispatch_targets]
        pred_values = [row[channel_index] for row in dispatch_preds]
        tp = sum(t == 1 and p == 1 for t, p in zip(target_values, pred_values))
        fp = sum(t == 0 and p == 1 for t, p in zip(target_values, pred_values))
        fn = sum(t == 1 and p == 0 for t, p in zip(target_values, pred_values))
        tn = sum(t == 0 and p == 0 for t, p in zip(target_values, pred_values))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        channel_metrics[channel_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_pos": tp + fn,
            "support_neg": tn + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        f1s.append(f1)
        recalls.append(recall)

    binary_tp = sum(t == 1 and p == 1 for t, p in zip(binary_targets, binary_preds))
    binary_fp = sum(t == 0 and p == 1 for t, p in zip(binary_targets, binary_preds))
    binary_fn = sum(t == 1 and p == 0 for t, p in zip(binary_targets, binary_preds))
    binary_tn = sum(t == 0 and p == 0 for t, p in zip(binary_targets, binary_preds))
    binary_precision = binary_tp / (binary_tp + binary_fp) if binary_tp + binary_fp else 0.0
    binary_recall = binary_tp / (binary_tp + binary_fn) if binary_tp + binary_fn else 0.0
    binary_f1 = (
        2 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if binary_precision + binary_recall
        else 0.0
    )

    dispatch_errors = sum(
        target != pred
        for target_row, pred_row in zip(dispatch_targets, dispatch_preds)
        for target, pred in zip(target_row, pred_row)
    )
    exact_matches = sum(
        target_row == pred_row
        for target_row, pred_row in zip(dispatch_targets, dispatch_preds)
    )
    dispatch_distribution = Counter(",".join(map(str, row)) for row in dispatch_preds)
    binary_distribution = Counter(binary_preds)

    return {
        "count_videos": count,
        "macro_f1": sum(f1s) / NUM_DISPATCH,
        "macro_recall": sum(recalls) / NUM_DISPATCH,
        "hamming_loss": dispatch_errors / (count * NUM_DISPATCH),
        "exact_match_ratio": exact_matches / count,
        "binary_auroc": float(roc_auc_score(binary_targets, binary_scores)),
        "binary_f1": binary_f1,
        "binary_accuracy": (binary_tp + binary_tn) / count,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        "binary_confusion": {
            "tp": binary_tp,
            "fp": binary_fp,
            "fn": binary_fn,
            "tn": binary_tn,
        },
        "dispatch_prediction_distribution": dict(sorted(dispatch_distribution.items())),
        "binary_prediction_distribution": {
            "0": binary_distribution[0],
            "1": binary_distribution[1],
        },
        "per_channel": channel_metrics,
    }


@torch.no_grad()
def run_eval(
    model: nn.Module,
    dataset: MilVideoDataset,
    sampler: FixedBagSampler,
    video_ids: List[str],
    device: torch.device,
    inject_teacher: bool,
) -> Dict:
    model.eval()

    all_targets = []       # [N, 3] int
    all_preds = []         # [N, 3] int (thresholded)
    all_binary_true = []   # [N] int
    all_binary_score = []  # [N] float
    all_binary_pred = []   # [N] int
    video_results = []

    bag_size = sampler.cfg.bag_size

    for vid in video_ids:
        clips = dataset.get_video_clips(vid)
        refs = sampler.select_refs_for_video(clips, epoch=0, video_id=vid)
        if len(refs) < bag_size:
            while len(refs) < bag_size:
                refs.append(refs[-1])
        refs = refs[:bag_size]

        bag = dataset.build_bag_from_refs(vid, refs)
        x = bag.embeddings
        if inject_teacher:
            ts = bag.teacher_scores.to(dtype=x.dtype).unsqueeze(1)
            x = torch.cat([x, ts], dim=1)

        x = x.unsqueeze(0).to(device)
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)

        out = model(x, mask)

        # Binary
        p_bin = torch.softmax(out.binary_logits, dim=1)
        bin_score = float(p_bin[0, 1].item())
        bin_pred = int(torch.argmax(out.binary_logits, dim=1).item())

        # Dispatch
        disp_scores = torch.sigmoid(out.anomaly_logits).squeeze(0).cpu().tolist()
        disp_pred = [1 if s >= PRIMARY_DISPATCH_THRESHOLD else 0 for s in disp_scores]

        info = dataset.video_labels[vid]
        disp_target = get_dispatch_label(str(info.group), int(info.label_bin))
        bin_true = 0 if int(info.label_bin) == 0 else 1

        all_targets.append(disp_target)
        all_preds.append(disp_pred)
        all_binary_true.append(bin_true)
        all_binary_score.append(bin_score)
        all_binary_pred.append(bin_pred)

        video_results.append({
            "video_id": vid,
            "binary_true": bin_true,
            "binary_score": bin_score,
            "binary_pred": bin_pred,
            "dispatch_target": disp_target,
            "dispatch_scores": disp_scores,
            "dispatch_pred": disp_pred,
        })

    metrics = compute_metrics(
        all_targets,
        all_preds,
        all_binary_true,
        all_binary_score,
        all_binary_pred,
    )
    metrics["per_video"] = video_results
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=os.environ.get("STAGE2_CFG"))
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--run_tag", required=True)
    args = parser.parse_args()

    split = args.split.strip().lower()
    run_tag = validate_run_tag(args.run_tag)
    if run_tag == "dispatch_v1":
        raise ValueError("dispatch_v1 is historical and protected from evaluation writes")

    run_root = Path("artifacts/stage2/runs") / run_tag
    cfg_path = run_root / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Resolved run configuration not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if bool(cfg["reproducibility"].get("validation_only", False)) and split != "val":
        raise ValueError("This run is validation-only; test evaluation is embargoed")
    evaluation_seed = int(cfg["reproducibility"]["evaluation_seed"])
    primary_threshold = float(cfg["reproducibility"]["primary_dispatch_threshold"])
    if primary_threshold != PRIMARY_DISPATCH_THRESHOLD:
        raise ValueError("The primary dispatch threshold must remain exactly 0.5")
    set_deterministic(evaluation_seed)

    out_dir = Path("artifacts/stage2/eval") / run_tag
    out_path = out_dir / f"dispatch_eval_{split}.json"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {out_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    store_root = cfg["paths"]["feature_store_root"]
    splits_root = cfg["paths"]["splits_root"]
    teacher_root = Path(cfg["paths"]["teacher_scores_root"])
    expected_dim = int(cfg["stage1"]["global_dim"])
    bag_size = int(cfg["stage2"]["bag_size"])

    split_ids = read_split_txt(str(Path(splits_root) / f"{split}.txt"))
    video_labels = load_videos_index(cfg["paths"]["videos_index_jsonl"])

    teacher = load_teacher_scores(str(teacher_root / f"teacher_scores_{split}.pt"))
    clip_lookup = build_clip_lookup_for_split(store_root, split, expected_dim, -1)
    vtc = build_video_to_clips_index(clip_lookup, teacher)
    dataset = MilVideoDataset(split_ids, vtc, video_labels, embedding_dim=expected_dim)

    source_identity = feature_store_identity(store_root, split)
    bag_manifest_path = manifest_path(
        cfg["reproducibility"]["evaluation_bag_manifest_pattern"],
        split,
        bag_size,
        evaluation_seed,
    )
    sampler = load_fixed_bag_sampler(
        path=bag_manifest_path,
        split=split,
        bag_size=bag_size,
        manifest_seed=evaluation_seed,
        video_ids=split_ids,
        video_to_clips=vtc,
        expected_source_identity=source_identity["identity_sha256"],
    )

    ckpt_path = resolve_checkpoint(run_tag)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    checkpoint_meta = ckpt.get("meta")
    if not isinstance(checkpoint_meta, dict):
        raise ValueError("Checkpoint metadata is missing")
    if checkpoint_meta.get("run_tag") != run_tag:
        raise ValueError("Checkpoint run_tag does not match the requested run tag")
    checkpoint_config_identity = checkpoint_meta.get("configuration_identity")
    if not checkpoint_config_identity:
        raise ValueError("Checkpoint configuration_identity is missing")
    resolved_config_identity = sha256_file(cfg_path)
    if checkpoint_config_identity != resolved_config_identity:
        raise ValueError("Checkpoint and resolved configuration identities do not match")
    inject_teacher = bool(checkpoint_meta.get("inject_teacher_score", True))
    input_dim = int(checkpoint_meta.get("input_dim", expected_dim + 1))

    model = HierarchicalAttentionMilHead(
        embedding_dim=input_dim,
        num_anomaly_classes=NUM_DISPATCH,
        attention_hidden_dim=int(cfg["stage2"].get("attention_hidden_dim", 256)),
        dropout=0.0,  # No dropout at eval
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    print(f"Evaluating dispatch model | split={split} | run={run_tag} | videos={len(split_ids)}")
    results = run_eval(model, dataset, sampler, split_ids, device, inject_teacher)
    results["provenance"] = {
        "run_tag": run_tag,
        "evaluation_seed": evaluation_seed,
        "bag_manifest": str(bag_manifest_path),
        "bag_manifest_sha256": sha256_file(bag_manifest_path),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "configuration_identity": resolved_config_identity,
        "threshold_policy": {
            "dispatch_primary": PRIMARY_DISPATCH_THRESHOLD,
            "mode": "fixed",
            "selection_procedure": None,
        },
    }

    # Print summary
    print(f"\n{'='*50}")
    print(f"Dispatch Eval Results ({split})")
    print(f"{'='*50}")
    print(f"Macro-F1:       {results['macro_f1']:.4f}")
    print(f"Macro Recall:   {results['macro_recall']:.4f}")
    print(f"Hamming Loss:   {results['hamming_loss']:.4f}")
    print(f"Exact Match:    {results['exact_match_ratio']:.4f}")
    print(f"Binary AUC-ROC: {results['binary_auroc']:.4f}")
    print(f"Binary F1:      {results['binary_f1']:.4f}")
    print(f"\nPer-Channel:")
    for ch, m in results["per_channel"].items():
        print(f"  {ch:10s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (pos={m['support_pos']})")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
