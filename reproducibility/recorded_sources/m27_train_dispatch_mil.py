"""
Stage 2 Dispatch MIL Training.

Architecture: HierarchicalAttentionMilHead with num_anomaly_classes=3
Binary head: Normal vs Anomaly (CE loss)
Dispatch head: [police, medical, fire] multi-label (ASL / BCE loss)

Usage:
    STAGE2_CFG=stage2_class/configs/hierA_dispatch_v1.yaml \
    python -m stage2_class.src.train.m27_train_dispatch_mil --run_tag dispatch_repro_seed1337
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from stage2_class.src.io.fixed_bag_manifest import (
    FixedBagSampler,
    load_fixed_bag_sampler,
    manifest_path,
    read_fixed_bag_manifest,
)
from stage2_class.src.io.training_bag_plan import (
    build_training_bag_plan,
    write_training_bag_plan,
)
from stage2_class.src.data.mg_mil_sampler import MgMilConfig, MgMilSampler
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
    NUM_DISPATCH,
    PRIMARY_DISPATCH_THRESHOLD,
    get_dispatch_label,
)
from stage2_class.src.utils.provenance import (
    feature_store_identity,
    git_state,
    hash_files,
    require_empty_run_directory,
    sha256_file,
    software_versions,
    validate_run_tag,
)
from stage2_class.src.utils.seed import set_deterministic


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification (Ridnik et al., 2021).
    Handles imbalance between positive and negative labels.
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B, C] in {0, 1}
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Asymmetric clipping (shift negatives)
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic BCE
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg

        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


from stage2_class.src.train.engine import (
    build_dispatch_bags_for_batch,
    canonical_group_name,
    collate_video_ids,
)


def compute_dispatch_metrics(
    all_targets: List[List[int]],
    all_preds: List[List[int]],
) -> Dict[str, float]:
    """Per-channel and aggregate metrics for multi-label dispatch."""
    from stage2_class.src.utils.dispatch_label_map import DISPATCH_CHANNELS

    targets = torch.tensor(all_targets, dtype=torch.int64)  # [N, 3]
    preds = torch.tensor(all_preds, dtype=torch.int64)      # [N, 3]

    metrics = {}
    f1s = []

    for ch in range(NUM_DISPATCH):
        tp = int(((targets[:, ch] == 1) & (preds[:, ch] == 1)).sum().item())
        fp = int(((targets[:, ch] == 0) & (preds[:, ch] == 1)).sum().item())
        fn = int(((targets[:, ch] == 1) & (preds[:, ch] == 0)).sum().item())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        name = DISPATCH_CHANNELS[ch]
        metrics[f"{name}_precision"] = prec
        metrics[f"{name}_recall"] = rec
        metrics[f"{name}_f1"] = f1
        f1s.append(f1)

    metrics["macro_f1"] = sum(f1s) / len(f1s)

    # Hamming loss
    hamming = float((targets != preds).float().mean().item())
    metrics["hamming_loss"] = hamming

    # Exact match ratio
    exact = float((targets == preds).all(dim=1).float().mean().item())
    metrics["exact_match_ratio"] = exact

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: MilVideoDataset,
    sampler: MgMilSampler | FixedBagSampler,
    video_ids: List[str],
    epoch: int,
    batch_size: int,
    device: torch.device,
    binary_weights: torch.Tensor,
    asl_loss_fn: nn.Module,
    inject_teacher_score: bool,
    dispatch_loss_weight: float,
) -> Dict:
    model.eval()
    binary_ce = nn.CrossEntropyLoss(weight=binary_weights)

    total_loss = 0.0
    binary_correct = 0
    all_dispatch_targets = []
    all_dispatch_preds = []
    all_binary_targets = []
    all_binary_preds = []
    n = 0

    for start in range(0, len(video_ids), batch_size):
        batch_ids = video_ids[start:start + batch_size]
        x, mask, y_bin, y_disp = build_dispatch_bags_for_batch(
            dataset, sampler, epoch, batch_ids, device, inject_teacher_score
        )

        out = model(x, mask)
        b_loss = binary_ce(out.binary_logits, y_bin)

        # Dispatch loss only on anomaly videos
        anom_mask = y_bin == 1
        if anom_mask.any():
            d_loss = asl_loss_fn(out.anomaly_logits[anom_mask], y_disp[anom_mask])
        else:
            d_loss = torch.tensor(0.0, device=device)

        loss = b_loss + dispatch_loss_weight * d_loss
        bs = y_bin.shape[0]
        total_loss += loss.item() * bs
        n += bs

        # Binary accuracy
        bin_pred = torch.argmax(out.binary_logits, dim=1)
        binary_correct += (bin_pred == y_bin).sum().item()
        all_binary_targets.extend(y_bin.cpu().tolist())
        all_binary_preds.extend(bin_pred.cpu().tolist())

        disp_pred = (torch.sigmoid(out.anomaly_logits) >= PRIMARY_DISPATCH_THRESHOLD).int()
        all_dispatch_targets.extend(y_disp.int().cpu().tolist())
        all_dispatch_preds.extend(disp_pred.cpu().tolist())

    dispatch_metrics = compute_dispatch_metrics(all_dispatch_targets, all_dispatch_preds)

    return {
        "loss": total_loss / max(1, n),
        "binary_accuracy": binary_correct / max(1, n),
        **dispatch_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", type=str, required=True)
    parser.add_argument(
        "--cfg",
        default=os.environ.get("STAGE2_CFG", "stage2_class/configs/hierA_dispatch_v1.yaml"),
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the YAML training seed for this run")
    parser.add_argument("--epochs", type=int, default=None, help="Override the YAML epoch limit for this run")
    args = parser.parse_args()

    cfg_path = Path(args.cfg)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    training_seed = int(args.seed if args.seed is not None else cfg.get("seed", 1337))
    evaluation_seed = int(cfg["reproducibility"]["evaluation_seed"])
    validation_only = bool(cfg["reproducibility"].get("validation_only", False))
    primary_threshold = float(cfg["reproducibility"]["primary_dispatch_threshold"])
    if primary_threshold != PRIMARY_DISPATCH_THRESHOLD:
        raise ValueError("The primary dispatch threshold must remain exactly 0.5")
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1")
        cfg["train"]["epochs"] = args.epochs
    cfg["seed"] = training_seed
    set_deterministic(training_seed)

    run_tag = validate_run_tag(args.run_tag)
    if run_tag == "dispatch_v1":
        raise ValueError("dispatch_v1 is historical and protected from training writes")
    run_root = Path("artifacts/stage2/runs") / run_tag
    require_empty_run_directory(run_root)

    store_root = cfg["paths"]["feature_store_root"]
    splits_root = Path(cfg["paths"]["splits_root"])
    videos_index_path = Path(cfg["paths"]["videos_index_jsonl"])
    teacher_root = Path(cfg["paths"]["teacher_scores_root"])
    expected_dim = int(cfg["stage1"]["global_dim"])
    bag_size = int(cfg["stage2"]["bag_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Run: {run_tag}")

    train_split_path = splits_root / "train.txt"
    val_split_path = splits_root / "val.txt"
    train_ids = read_split_txt(str(train_split_path))
    val_ids = read_split_txt(str(val_split_path))
    video_labels = load_videos_index(str(videos_index_path))

    teacher_paths = {
        split: teacher_root / f"teacher_scores_{split}.pt"
        for split in (("train", "val") if validation_only else ("train", "val", "test"))
    }
    teacher_train = load_teacher_scores(str(teacher_paths["train"]))
    teacher_val = load_teacher_scores(str(teacher_paths["val"]))

    clip_train = build_clip_lookup_for_split(store_root, "train", expected_dim, -1)
    clip_val = build_clip_lookup_for_split(store_root, "val", expected_dim, -1)
    vtc_train = build_video_to_clips_index(clip_train, teacher_train)
    vtc_val = build_video_to_clips_index(clip_val, teacher_val)

    feature_identities = {
        split: feature_store_identity(store_root, split)
        for split in (("train", "val") if validation_only else ("train", "val", "test"))
    }
    manifest_pattern = cfg["reproducibility"]["evaluation_bag_manifest_pattern"]
    val_manifest_path = manifest_path(manifest_pattern, "val", bag_size, evaluation_seed)
    val_sampler = load_fixed_bag_sampler(
        path=val_manifest_path,
        split="val",
        bag_size=bag_size,
        manifest_seed=evaluation_seed,
        video_ids=val_ids,
        video_to_clips=vtc_val,
        expected_source_identity=feature_identities["val"]["identity_sha256"],
    )
    if not validation_only:
        test_split_path = splits_root / "test.txt"
        test_ids = read_split_txt(str(test_split_path))
        test_manifest_path = manifest_path(manifest_pattern, "test", bag_size, evaluation_seed)
        read_fixed_bag_manifest(
            path=test_manifest_path,
            split="test",
            bag_size=bag_size,
            manifest_seed=evaluation_seed,
            video_ids=test_ids,
            expected_source_identity=feature_identities["test"]["identity_sha256"],
        )

    run_ckpt = run_root / "checkpoints"
    run_train = run_root / "train"
    run_ckpt.mkdir(parents=True, exist_ok=True)
    run_train.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_root / "resolved_config.yaml"
    with resolved_config_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    config_identity = sha256_file(resolved_config_path)

    patience = int(cfg["train"].get("early_stop_patience", 12))
    min_delta = float(cfg["train"].get("early_stop_min_delta", 1e-4))
    epochs = int(cfg["train"]["epochs"])
    input_files = [
        cfg_path,
        videos_index_path,
        train_split_path,
        val_split_path,
        teacher_paths["train"],
        teacher_paths["val"],
        val_manifest_path,
    ]
    if not validation_only:
        input_files.extend([test_split_path, teacher_paths["test"], test_manifest_path])
    run_manifest = {
        "run_tag": run_tag,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "early_stopping": {
            "patience": patience,
            "minimum_delta": min_delta,
            "maximum_epochs": epochs,
            "selected_epoch": None,
        },
        "bag_settings": {
            "bag_size": bag_size,
            "warmup_epochs": int(cfg["stage2"]["warmup_epochs"]),
            "mixture_uniform_ratio": float(cfg["stage2"]["mixture_uniform_ratio"]),
            "top_fraction": float(cfg["stage2"]["top_fraction"]),
            "validation": "fixed_manifest",
            "test": "embargoed_not_loaded" if validation_only else "fixed_manifest",
        },
        "threshold_policy": {
            "dispatch_primary": PRIMARY_DISPATCH_THRESHOLD,
            "mode": "fixed",
            "selection_procedure": None,
        },
        "configuration_identity": config_identity,
        "git": git_state(),
        "device": {
            "requested": cfg.get("device"),
            "resolved": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "software_versions": software_versions(),
        "input_paths": {
            "source_config": str(cfg_path),
            "resolved_config": str(resolved_config_path),
            "feature_store_root": str(store_root),
            "splits_root": str(splits_root),
            "videos_index": str(videos_index_path),
            "teacher_scores": {split: str(path) for split, path in teacher_paths.items()},
            "evaluation_bag_manifests": {"val": str(val_manifest_path)},
        },
        "artifact_hashes": {
            "files": hash_files(input_files),
            "resolved_config": config_identity,
            "feature_store": {
                split: {
                    "identity_sha256": identity["identity_sha256"],
                    "file_count": len(identity["files"]),
                }
                for split, identity in feature_identities.items()
            },
        },
        "reproducibility_scope": cfg["reproducibility"]["feature_store_scope"],
        "validation_only": validation_only,
        "execution": {
            "command": shlex.join([sys.executable, *sys.argv]),
            "working_directory": str(Path.cwd()),
        },
    }
    if not validation_only:
        run_manifest["input_paths"]["evaluation_bag_manifests"]["test"] = str(test_manifest_path)
    run_manifest_path = run_root / "run_manifest.json"
    with run_manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # Class weights for binary head
    train_binary = [0 if int(video_labels[v].label_bin) == 0 else 1 for v in train_ids]
    counts = Counter(train_binary)
    total = len(train_binary)
    bw = torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float32)
    binary_weights = (bw / bw.mean()).to(device)
    print(f"Binary weights: {binary_weights.tolist()}")

    train_ds = MilVideoDataset(train_ids, vtc_train, video_labels, embedding_dim=expected_dim)
    val_ds = MilVideoDataset(val_ids, vtc_val, video_labels, embedding_dim=expected_dim)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size_videos"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=bool(cfg["train"]["pin_memory"]),
        collate_fn=collate_video_ids,
        generator=torch.Generator().manual_seed(training_seed),
    )

    sampler_cfg = MgMilConfig(
        bag_size=bag_size,
        warmup_epochs=int(cfg["stage2"]["warmup_epochs"]),
        mixture_uniform_ratio=float(cfg["stage2"]["mixture_uniform_ratio"]),
        top_fraction=float(cfg["stage2"]["top_fraction"]),
    )
    train_sampler = MgMilSampler(cfg=sampler_cfg, seed=training_seed)

    training_bag_plan = build_training_bag_plan(
        video_ids=train_ids,
        video_to_clips=vtc_train,
        sampler=train_sampler,
        maximum_epochs=epochs,
        source_feature_identity=feature_identities["train"]["identity_sha256"],
        teacher_scores_sha256=sha256_file(teacher_paths["train"]),
    )
    training_bag_plan_path = run_root / "training_bag_plan.json"
    write_training_bag_plan(training_bag_plan_path, training_bag_plan)
    run_manifest["training_bag_plan"] = {
        "path": str(training_bag_plan_path),
        "file_sha256": sha256_file(training_bag_plan_path),
        "plan_identity_sha256": training_bag_plan["plan_identity_sha256"],
        "training_order_seed": training_seed,
    }
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    inject_teacher = bool(cfg["stage2"].get("inject_teacher_score", True))
    input_dim = expected_dim + (1 if inject_teacher else 0)

    # Model — reuse existing architecture, just 3 outputs on dispatch head
    model = HierarchicalAttentionMilHead(
        embedding_dim=input_dim,
        num_anomaly_classes=NUM_DISPATCH,
        attention_hidden_dim=int(cfg["stage2"].get("attention_hidden_dim", 256)),
        dropout=float(cfg["stage2"].get("dropout", 0.1)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    binary_ce = nn.CrossEntropyLoss(weight=binary_weights)
    asl_loss = AsymmetricLoss(
        gamma_neg=float(cfg["train"].get("asl_gamma_neg", 4.0)),
        gamma_pos=float(cfg["train"].get("asl_gamma_pos", 0.0)),
        clip=float(cfg["train"].get("asl_clip", 0.05)),
    )

    grad_clip = float(cfg["train"]["grad_clip_norm"])
    dispatch_loss_weight = float(cfg["train"].get("dispatch_loss_weight", 1.0))

    best_val_f1 = -1.0
    best_epoch = None
    no_improve = 0
    log_path = run_train / "dispatch_train_log.jsonl"

    print(f"Training dispatch MIL | epochs={epochs} | dispatch_weight={dispatch_loss_weight}")
    print(f"ASL: gamma_neg={asl_loss.gamma_neg}, gamma_pos={asl_loss.gamma_pos}, clip={asl_loss.clip}")

    with log_path.open("x", encoding="utf-8") as log_file:
        for epoch in range(epochs):
            model.train()
            t0 = time.time()
            running_loss = 0.0
            n_videos = 0

            for batch in train_loader:
                x, mask, y_bin, y_disp = build_dispatch_bags_for_batch(
                    train_ds, train_sampler, epoch, batch["video_ids"], device, inject_teacher
                )

                out = model(x, mask)
                b_loss = binary_ce(out.binary_logits, y_bin)

                anom_mask = y_bin == 1
                if anom_mask.any():
                    d_loss = asl_loss(out.anomaly_logits[anom_mask], y_disp[anom_mask])
                else:
                    d_loss = torch.tensor(0.0, device=device)

                loss = b_loss + dispatch_loss_weight * d_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                bs = y_bin.shape[0]
                running_loss += loss.item() * bs
                n_videos += bs

            val_m = evaluate(
                model, val_ds, val_sampler, val_ids, 0,
                int(cfg["train"]["batch_size_videos"]), device,
                binary_weights, asl_loss, inject_teacher, dispatch_loss_weight
            )

            record = {
                "epoch": epoch,
                "train_loss": running_loss / max(1, n_videos),
                "time_sec": time.time() - t0,
                **{f"val_{k}": v for k, v in val_m.items()},
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            print(f"Ep {epoch:3d} | train_loss={record['train_loss']:.4f} | "
                  f"val_macro_f1={val_m['macro_f1']:.4f} | val_hamming={val_m['hamming_loss']:.4f} | "
                  f"{time.time()-t0:.1f}s")

            if val_m["macro_f1"] > best_val_f1 + min_delta:
                best_val_f1 = val_m["macro_f1"]
                best_epoch = epoch
                no_improve = 0
                torch.save({
                    "meta": {
                        "model_type": "dispatch_mil",
                        "epoch": epoch,
                        "selected_epoch": epoch,
                        "best_val_macro_f1": best_val_f1,
                        "input_dim": input_dim,
                        "num_dispatch": NUM_DISPATCH,
                        "inject_teacher_score": inject_teacher,
                        "run_tag": run_tag,
                        "training_seed": training_seed,
                        "evaluation_seed": evaluation_seed,
                        "early_stop_patience": patience,
                        "early_stop_min_delta": min_delta,
                        "maximum_epochs": epochs,
                        "configuration_identity": config_identity,
                        "validation_bag_manifest_sha256": sha256_file(val_manifest_path),
                        "primary_dispatch_threshold": PRIMARY_DISPATCH_THRESHOLD,
                    },
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, run_ckpt / "mil_best.pt")
                print(f"  -> Saved best (F1={best_val_f1:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stop at epoch {epoch}. Best F1={best_val_f1:.4f}")
                    break

    # Final summary
    summary = {
        "best_val_macro_f1": best_val_f1,
        "selected_epoch": best_epoch,
        "epochs_ran": epoch + 1,
        "run_tag": run_tag,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "configuration_identity": config_identity,
    }
    summary_path = run_train / "dispatch_summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    run_manifest["early_stopping"]["selected_epoch"] = best_epoch
    run_manifest["best_val_macro_f1"] = best_val_f1
    run_manifest["epochs_ran"] = epoch + 1
    run_manifest["output_hashes"] = hash_files([
        run_ckpt / "mil_best.pt",
        log_path,
        summary_path,
    ])
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Done. Best dispatch Macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()
