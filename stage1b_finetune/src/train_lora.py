# stage1b_finetune/src/train_lora.py
"""
Stage 1b: LoRA fine-tuning of VideoMAE-Base on UCF-Crime.

Clip-level training with:
- Teacher-score weighted cross-entropy loss
- Mixed precision (FP16)
- Gradient checkpointing
- Video-centric iteration with gradient accumulation

Usage:
    python -m stage1b_finetune.src.train_lora --cfg stage1b_finetune/configs/lora_finetune.yaml

Hardware target: RTX 4050 (6 GB VRAM), 24 GB RAM
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import yaml

from stage1b_finetune.src.lora_backbone import LoRAVideoMAE
from stage1b_finetune.src.clip_dataset import (
    ClipLevelDataset,
    ClipSample,
    build_canonical_label_map,
    load_teacher_scores_lookup,
    load_videos_index,
    read_split_txt,
)


def set_deterministic(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted cross-entropy loss with per-sample teacher weights.

    Args:
        logits: [B, C]
        labels: [B] int64
        weights: [B] float32 per-clip teacher weights
        class_weights: [C] float32 class imbalance weights
    """
    # Per-sample CE (no reduction)
    ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    # Apply teacher score weights
    weighted_loss = ce_loss * weights
    return weighted_loss.mean()


def make_class_weights(label_map: Dict[str, int], video_ids: List[str], videos_index) -> torch.Tensor:
    """Compute class weights from training distribution."""
    num_classes = len(label_map)
    counts = Counter()

    for vid in video_ids:
        rec = videos_index[vid]
        if rec.label_bin == 0:
            counts[label_map["Normal"]] += 1
        else:
            counts[label_map[rec.group]] += 1

    total = len(video_ids)
    weights = []
    for c in range(num_classes):
        count_c = counts.get(c, 0)
        if count_c == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * float(count_c)))

    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.mean().clamp_min(1e-12)
    return w


@torch.no_grad()
def evaluate_clip_level(
    model: LoRAVideoMAE,
    dataset: ClipLevelDataset,
    device: torch.device,
    class_weights: torch.Tensor,
    epoch: int,
    max_videos: int = 50,
) -> Dict:
    """
    Quick clip-level evaluation on validation set.
    Evaluates a subset of videos for speed.
    """
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0
    pred_counter = Counter()
    all_preds = []
    all_targets = []

    videos_seen = 0
    for video_clips in dataset.iterate_epoch(epoch=epoch):
        if videos_seen >= max_videos:
            break
        videos_seen += 1

        for clip_sample in video_clips:
            x = clip_sample.clip_tensor.unsqueeze(0).to(device)
            label = torch.tensor([clip_sample.label], dtype=torch.int64, device=device)

            with autocast(device_type="cuda", dtype=torch.float16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits, label, weight=class_weights)

            pred = int(torch.argmax(logits, dim=1).item())
            target = clip_sample.label

            correct += 1 if pred == target else 0
            total += 1
            loss_sum += float(loss.item())
            pred_counter.update([pred])
            all_preds.append(pred)
            all_targets.append(target)

    # Compute macro metrics
    num_classes = len(class_weights)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for t, p in zip(all_targets, all_preds):
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    recalls = []
    for c in range(num_classes):
        denom = tp[c] + fn[c]
        recalls.append(tp[c] / denom if denom > 0 else 0.0)

    return {
        "accuracy": correct / max(1, total),
        "avg_loss": loss_sum / max(1, total),
        "macro_recall": sum(recalls) / num_classes,
        "total_clips": total,
        "videos_evaluated": videos_seen,
        "pred_dist": dict(pred_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1b: LoRA Fine-tuning")
    parser.add_argument(
        "--cfg", type=str,
        default=os.environ.get("LORA_CFG", "stage1b_finetune/configs/lora_finetune.yaml"),
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())

    seed = int(cfg["seed"])
    set_deterministic(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Paths
    raw_root = cfg["paths"]["raw_video_root"]
    videos_jsonl = cfg["paths"]["videos_jsonl"]
    splits_root = cfg["paths"]["splits_root"]
    teacher_dir = cfg["paths"]["teacher_scores_dir"]
    ckpt_dir = cfg["paths"]["lora_checkpoint_dir"]
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading videos index...")
    videos_index = load_videos_index(videos_jsonl)

    train_video_ids = read_split_txt(str(Path(splits_root) / "train.txt"))
    val_video_ids = read_split_txt(str(Path(splits_root) / "val.txt"))

    # Filter to videos that exist in index
    train_video_ids = [v for v in train_video_ids if v in videos_index]
    val_video_ids = [v for v in val_video_ids if v in videos_index]
    print(f"Train videos: {len(train_video_ids)}, Val videos: {len(val_video_ids)}")

    # Build label map (must match Stage-2)
    label_map = build_canonical_label_map(videos_index, train_video_ids)
    num_classes = cfg["num_classes"]
    if len(label_map) != num_classes:
        raise ValueError(f"Label map has {len(label_map)} classes, expected {num_classes}")
    print(f"Label map: {label_map}")

    # Save label map for reference
    label_map_path = Path(ckpt_dir) / "label_map.json"
    label_map_path.write_text(json.dumps(label_map, indent=2))

    # Load teacher scores
    print("Loading teacher scores...")
    teacher_train = load_teacher_scores_lookup(str(Path(teacher_dir) / "teacher_scores_train.pt"))
    teacher_val = load_teacher_scores_lookup(str(Path(teacher_dir) / "teacher_scores_val.pt"))
    print(f"Teacher scores: train={len(teacher_train)}, val={len(teacher_val)}")

    # Build datasets
    train_cfg = cfg["train"]
    train_dataset = ClipLevelDataset(
        raw_root=raw_root,
        videos_index=videos_index,
        video_ids=train_video_ids,
        label_map=label_map,
        teacher_scores=teacher_train,
        clips_per_video=int(train_cfg["clips_per_video"]),
        clip_len=int(cfg["backbone"]["clip_len"]),
        img_size=int(cfg["backbone"]["img_size"]),
        teacher_weight_temperature=float(train_cfg["teacher_weight_temperature"]),
        normal_clip_weight=float(train_cfg["normal_clip_weight"]),
        min_anomaly_clip_weight=float(train_cfg["min_anomaly_clip_weight"]),
        seed=seed,
    )

    val_dataset = ClipLevelDataset(
        raw_root=raw_root,
        videos_index=videos_index,
        video_ids=val_video_ids,
        label_map=label_map,
        teacher_scores=teacher_val,
        clips_per_video=int(train_cfg["clips_per_video"]),
        clip_len=int(cfg["backbone"]["clip_len"]),
        img_size=int(cfg["backbone"]["img_size"]),
        teacher_weight_temperature=float(train_cfg["teacher_weight_temperature"]),
        normal_clip_weight=float(train_cfg["normal_clip_weight"]),
        min_anomaly_clip_weight=float(train_cfg["min_anomaly_clip_weight"]),
        seed=seed,
    )

    # Class weights for imbalanced training
    class_weights = make_class_weights(label_map, train_video_ids, videos_index).to(device)
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # Build model
    print("Building LoRA model...")
    lora_cfg = cfg["lora"]
    model = LoRAVideoMAE(
        num_classes=num_classes,
        lora_rank=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=lora_cfg["target_modules"],
        use_gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
    ).to(device)

    # Optimizer (only LoRA params + classifier are trainable)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    # Mixed precision scaler
    scaler = GradScaler(device="cuda", enabled=bool(train_cfg["fp16"]))

    # Training config
    epochs = int(train_cfg["epochs"])
    batch_size_clips = int(train_cfg["batch_size_clips"])
    grad_accum_videos = int(train_cfg["gradient_accumulation_videos"])
    grad_clip_norm = float(train_cfg["grad_clip_norm"])

    # LR scheduler (linear warmup + cosine decay)
    total_videos_per_epoch = len(train_video_ids)
    # Ceil division: accounts for the final partial-accumulation step each epoch
    steps_per_epoch = -(-total_videos_per_epoch // grad_accum_videos)
    total_steps = steps_per_epoch * epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(train_cfg["learning_rate"]),
        total_steps=total_steps,
        pct_start=float(train_cfg["warmup_ratio"]),
        anneal_strategy="cos",
        div_factor=10.0,          # start_lr = max_lr / 10
        final_div_factor=100.0,   # end_lr = start_lr / 100
    )

    # Training log
    log_path = Path(ckpt_dir) / "train_log.jsonl"
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Starting LoRA Fine-tuning")
    print(f"  Epochs: {epochs}")
    print(f"  Clips per video: {train_cfg['clips_per_video']}")
    print(f"  Clip batch size: {batch_size_clips}")
    print(f"  Gradient accumulation: {grad_accum_videos} videos")
    print(f"  Effective batch: {grad_accum_videos * train_cfg['clips_per_video']} clips/step")
    print(f"  Total optimizer steps: ~{total_steps}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        videos_processed = 0
        optimizer_steps = 0

        optimizer.zero_grad(set_to_none=True)

        for video_clips in train_dataset.iterate_epoch(epoch=epoch):
            videos_processed += 1

            # Process clips from this video in mini-batches
            for i in range(0, len(video_clips), batch_size_clips):
                batch = video_clips[i:i + batch_size_clips]

                # Stack batch
                x = torch.stack([c.clip_tensor for c in batch]).to(device)
                labels = torch.tensor([c.label for c in batch], dtype=torch.int64, device=device)
                weights = torch.tensor(
                    [c.teacher_weight for c in batch], dtype=torch.float32, device=device
                )

                # Forward + loss (mixed precision)
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits, _ = model(x)
                    loss = compute_weighted_ce_loss(logits, labels, weights, class_weights)
                    # Scale by accumulation factor
                    loss = loss / float(grad_accum_videos)

                # Backward (scaled)
                scaler.scale(loss).backward()

                # Track metrics
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    running_correct += int((preds == labels).sum().item())
                    running_total += len(batch)
                    running_loss += float(loss.item()) * grad_accum_videos

            # Optimizer step after accumulation
            if videos_processed % grad_accum_videos == 0:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1

            # Progress logging
            if videos_processed % 50 == 0:
                lr_current = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch} | Videos: {videos_processed}/{len(train_video_ids)} | "
                    f"Loss: {running_loss/max(1,running_total):.4f} | "
                    f"Acc: {running_correct/max(1,running_total):.3f} | "
                    f"LR: {lr_current:.2e}"
                )

        # Final optimizer step for remaining gradients
        if videos_processed % grad_accum_videos != 0:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if total_steps > 0:
                scheduler.step()

        train_acc = running_correct / max(1, running_total)
        train_loss = running_loss / max(1, running_total)
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch} Training Complete:")
        print(f"  Time: {epoch_time:.1f}s | Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")
        print(f"  Videos: {videos_processed} | Clips: {running_total} | Steps: {optimizer_steps}")

        # Validation
        print("  Running validation...")
        val_metrics = evaluate_clip_level(
            model=model,
            dataset=val_dataset,
            device=device,
            class_weights=class_weights,
            epoch=epoch,
            max_videos=min(50, len(val_video_ids)),
        )
        print(
            f"  Val: Acc={val_metrics['accuracy']:.3f} | "
            f"Loss={val_metrics['avg_loss']:.4f} | "
            f"MacroRecall={val_metrics['macro_recall']:.3f} | "
            f"Clips={val_metrics['total_clips']}"
        )

        # Log
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_clips": running_total,
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_metrics["avg_loss"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_pred_dist": val_metrics["pred_dist"],
            "epoch_time_sec": epoch_time,
            "optimizer_steps": optimizer_steps,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Save checkpoint
        ckpt_path = model.save_lora_checkpoint(
            save_dir=ckpt_dir,
            epoch=epoch,
            extra_meta={
                "train_accuracy": train_acc,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_recall": val_metrics["macro_recall"],
                "label_map": label_map,
                "cfg_path": args.cfg,
            },
        )
        print(f"  Saved: {ckpt_path}")

        # Track best
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_path = Path(ckpt_dir) / "lora_best.pt"
            # Copy best checkpoint
            import shutil
            shutil.copy2(ckpt_path, str(best_path))
            print(f"  New best val accuracy: {best_val_acc:.4f} -> {best_path}")

        print()

    print(f"\n{'='*60}")
    print(f"Training Complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
