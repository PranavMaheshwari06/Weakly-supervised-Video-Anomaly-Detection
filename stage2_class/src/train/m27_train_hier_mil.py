from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

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
from stage2_class.src.utils.label_map import save_label_map
from stage2_class.src.utils.seed import set_deterministic



class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight # Class weights tensor [C]
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def canonical_group_name(group: str, label_bin: int) -> str:
    """
    UCF-Crime uses different normal group names across splits.
    For hierarchical training we must force one canonical Normal class.
    """
    if int(label_bin) == 0:
        return "Normal"
    return str(group)


def build_canonical_label_map_from_train(
    train_video_ids: List[str],
    video_labels: Dict[str, object],
) -> Dict[str, int]:
    """
    Build a stable class map with:
      Normal -> 0
      anomaly classes -> 1..K
    """
    group_set = set()

    for video_id in train_video_ids:
        info = video_labels[video_id]
        group_name = canonical_group_name(info.group, info.label_bin)
        group_set.add(group_name)

    class_names = ["Normal"] + sorted([group_name for group_name in group_set if group_name != "Normal"])
    label_map = {group_name: class_id for class_id, group_name in enumerate(class_names)}
    return label_map


def collate_video_ids(batch: List[Tuple[str, int, str]]) -> Dict:
    video_ids = [item[0] for item in batch]
    groups = [item[2] for item in batch]
    label_bin = [int(item[1]) for item in batch]
    return {
        "video_ids": video_ids,
        "groups": groups,
        "label_bin": label_bin,
    }


def make_class_weights(class_ids: List[int], num_classes: int) -> torch.Tensor:
    counts = Counter(class_ids)
    total = len(class_ids)

    weights: List[float] = []
    for class_id in range(num_classes):
        count_value = counts.get(class_id, 0)
        if count_value == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * float(count_value)))

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    weight_tensor = weight_tensor / weight_tensor.mean().clamp_min(1e-12)
    return weight_tensor


def compute_macro_metrics(
    all_targets: List[int],
    all_preds: List[int],
    num_classes: int,
) -> Dict[str, float]:
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for target_value, pred_value in zip(all_targets, all_preds):
        if pred_value == target_value:
            tp[target_value] += 1
        else:
            fp[pred_value] += 1
            fn[target_value] += 1

    recalls: List[float] = []
    f1s: List[float] = []

    for class_id in range(num_classes):
        denom_recall = tp[class_id] + fn[class_id]
        denom_precision = tp[class_id] + fp[class_id]

        recall = tp[class_id] / denom_recall if denom_recall > 0 else 0.0
        precision = tp[class_id] / denom_precision if denom_precision > 0 else 0.0

        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        recalls.append(recall)
        f1s.append(f1)

    return {
        "macro_recall": sum(recalls) / float(num_classes),
        "macro_f1": sum(f1s) / float(num_classes),
    }


def build_hierarchical_maps(label_map: Dict[str, int]) -> Dict:
    class_names = [group_name for group_name, _ in sorted(label_map.items(), key=lambda x: x[1])]

    if len(class_names) < 2:
        raise ValueError("Need at least Normal + 1 anomaly class")

    if label_map.get("Normal", None) != 0:
        raise ValueError("Expected Normal to map to class id 0")

    anomaly_class_names = class_names[1:]
    full_to_anomaly: Dict[int, int] = {}
    anomaly_to_full: Dict[int, int] = {}

    for anomaly_index, full_class_id in enumerate(range(1, len(class_names))):
        full_to_anomaly[full_class_id] = anomaly_index
        anomaly_to_full[anomaly_index] = full_class_id

    return {
        "class_names": class_names,
        "anomaly_class_names": anomaly_class_names,
        "full_to_anomaly": full_to_anomaly,
        "anomaly_to_full": anomaly_to_full,
        "num_full_classes": len(class_names),
        "num_anomaly_classes": len(anomaly_class_names),
    }


def build_bags_for_batch(
    dataset: MilVideoDataset,
    sampler: MgMilSampler,
    label_map: Dict[str, int],
    epoch: int,
    video_ids: List[str],
    device: torch.device,
    inject_teacher_score: bool,
    full_to_anomaly: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      x_batch:           [B, M, D]
      mask_batch:        [B, M]
      y_full_batch:      [B]    0..13
      y_binary_batch:    [B]    0=Normal, 1=Anomaly
      y_anomaly_batch:   [B]   -1 for normal, else 0..12
    """
    bag_embeddings: List[torch.Tensor] = []
    bag_masks: List[torch.Tensor] = []
    bag_full_targets: List[int] = []
    bag_binary_targets: List[int] = []
    bag_anomaly_targets: List[int] = []

    bag_size = sampler.cfg.bag_size

    for video_id in video_ids:
        clips = dataset.get_video_clips(video_id)
        refs = sampler.select_refs_for_video(clips, epoch=epoch, video_id=video_id)

        if len(refs) < bag_size:
            while len(refs) < bag_size:
                refs.append(refs[-1])

        refs = refs[:bag_size]
        bag = dataset.build_bag_from_refs(video_id, refs)

        x = bag.embeddings
        mask = torch.ones(x.shape[0], dtype=torch.bool)

        if inject_teacher_score:
            teacher_scores = bag.teacher_scores.to(dtype=x.dtype).unsqueeze(1)
            x = torch.cat([x, teacher_scores], dim=1)

        group_name = canonical_group_name(bag.group, bag.label_bin)
        if group_name not in label_map:
            raise ValueError(f"Group '{group_name}' missing in label_map")

        full_class_id = int(label_map[group_name])
        binary_target = 0 if full_class_id == 0 else 1
        anomaly_target = -1 if full_class_id == 0 else int(full_to_anomaly[full_class_id])

        bag_embeddings.append(x)
        bag_masks.append(mask)
        bag_full_targets.append(full_class_id)
        bag_binary_targets.append(binary_target)
        bag_anomaly_targets.append(anomaly_target)

    x_batch = torch.stack(bag_embeddings, dim=0).to(device)
    mask_batch = torch.stack(bag_masks, dim=0).to(device)
    y_full_batch = torch.tensor(bag_full_targets, dtype=torch.int64, device=device)
    y_binary_batch = torch.tensor(bag_binary_targets, dtype=torch.int64, device=device)
    y_anomaly_batch = torch.tensor(bag_anomaly_targets, dtype=torch.int64, device=device)

    return x_batch, mask_batch, y_full_batch, y_binary_batch, y_anomaly_batch


def compose_full_preds_from_logits(
    binary_logits: torch.Tensor,
    anomaly_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Keep probabilities for evaluation/logging
    full_probs = HierarchicalAttentionMilHead.compose_full_probs(binary_logits, anomaly_logits)
    
    # cascaded decision: if binary head says normal, force class 0
    bin_preds = torch.argmax(binary_logits, dim=1)           # 0 or 1
    anom_preds = torch.argmax(anomaly_logits, dim=1) + 1     # 1..13
    full_preds = torch.where(bin_preds == 1, anom_preds, torch.zeros_like(bin_preds))
    
    return full_probs, full_preds

def compute_full_loss_from_probs(
    full_probs: torch.Tensor,
    y_full: torch.Tensor,
) -> torch.Tensor:
    """
    full_probs: [B, C] probabilities that already sum to 1
    y_full: [B] full class ids in 0..C-1

    We use NLL on composed probabilities so the final 14-class decision
    is trained directly, not only indirectly through the two heads.
    """
    full_probs = full_probs.clamp_min(1e-8)
    log_full_probs = torch.log(full_probs)
    loss = F.nll_loss(log_full_probs, y_full)
    return loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: MilVideoDataset,
    sampler: MgMilSampler,
    label_map: Dict[str, int],
    video_ids: List[str],
    epoch: int,
    batch_size_videos: int,
    device: torch.device,
    binary_class_weights: torch.Tensor,
    anomaly_class_weights: torch.Tensor,
    num_full_classes: int,
    inject_teacher_score: bool,
    full_to_anomaly: Dict[int, int],
    anomaly_loss_weight: float,
) -> Dict:
    model.eval()

    binary_ce = nn.CrossEntropyLoss(weight=binary_class_weights)
    anomaly_ce = FocalLoss(weight=anomaly_class_weights, gamma=2.0)

    total_loss_sum = 0.0
    binary_loss_sum = 0.0
    anomaly_loss_sum = 0.0
    full_loss_sum = 0.0

    total_videos = 0
    correct_full = 0

    pred_counter: Counter = Counter()
    all_targets: List[int] = []
    all_preds: List[int] = []

    for start_index in range(0, len(video_ids), batch_size_videos):
        batch_video_ids = video_ids[start_index : start_index + batch_size_videos]

        x, mask, y_full, y_binary, y_anomaly = build_bags_for_batch(
            dataset=dataset,
            sampler=sampler,
            label_map=label_map,
            epoch=epoch,
            video_ids=batch_video_ids,
            device=device,
            inject_teacher_score=inject_teacher_score,
            full_to_anomaly=full_to_anomaly,
        )

        outputs = model(x, mask)
        binary_loss = binary_ce(outputs.binary_logits, y_binary)

        anomaly_mask = y_binary == 1
        if bool(anomaly_mask.any().item()):
            anomaly_loss = anomaly_ce(outputs.anomaly_logits[anomaly_mask], y_anomaly[anomaly_mask])
        else:
            anomaly_loss = torch.tensor(0.0, device=device)

        full_probs, full_preds = compose_full_preds_from_logits(
            outputs.binary_logits,
            outputs.anomaly_logits,
        )
        full_loss = compute_full_loss_from_probs(full_probs, y_full)

        total_loss = binary_loss + anomaly_loss_weight * anomaly_loss

        pred_list = full_preds.detach().cpu().tolist()
        target_list = y_full.detach().cpu().tolist()

        pred_counter.update(pred_list)
        all_preds.extend(pred_list)
        all_targets.extend(target_list)

        batch_count = int(y_full.shape[0])
        total_videos += batch_count
        correct_full += int((full_preds == y_full).sum().item())

        total_loss_sum += float(total_loss.item()) * batch_count
        binary_loss_sum += float(binary_loss.item()) * batch_count
        anomaly_loss_sum += float(anomaly_loss.item()) * batch_count
        full_loss_sum += float(full_loss.item()) * batch_count

    macro_metrics = compute_macro_metrics(all_targets, all_preds, num_full_classes)

    return {
        "loss": total_loss_sum / max(1, total_videos),
        "binary_loss": binary_loss_sum / max(1, total_videos),
        "anomaly_loss": anomaly_loss_sum / max(1, total_videos),
        "full_loss": full_loss_sum / max(1, total_videos),
        "accuracy": correct_full / max(1, total_videos),
        "macro_recall": macro_metrics["macro_recall"],
        "macro_f1": macro_metrics["macro_f1"],
        "count_videos": total_videos,
        "pred_class_dist": dict(pred_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 Hierarchical MIL Training")
    parser.add_argument(
        "--run_tag",
        type=str,
        default=os.environ.get("RUN_TAG", "hier_stage2_run"),
        help="Unique run tag for organizing outputs",
    )
    args = parser.parse_args()

    cfg_path = os.environ.get("STAGE2_CFG", "stage2_class/configs/hierA_dual_head.yaml")
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    seed = int(cfg.get("seed", 1337))
    set_deterministic(seed)

    run_tag = str(args.run_tag).strip()
    run_root = Path("artifacts/stage2/runs") / run_tag
    run_ckpt_dir = run_root / "checkpoints"
    run_train_dir = run_root / "train"
    run_label_map_dir = run_root / "label_map"

    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_train_dir.mkdir(parents=True, exist_ok=True)
    run_label_map_dir.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("train", {})
    cfg["train"]["save_dir_checkpoints"] = str(run_ckpt_dir)

    cfg.setdefault("io", {})
    cfg["io"]["write_dir_results"] = str(run_train_dir)
    cfg["io"]["write_dir_logs"] = str(run_train_dir)

    store_root = cfg["paths"]["feature_store_root"]
    splits_root = cfg["paths"]["splits_root"]
    videos_index_path = cfg["paths"]["videos_index_jsonl"]
    expected_dim = int(cfg["stage1"]["global_dim"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    )
    print(f"Device: {device}")
    print(f"Run tag: {run_tag}")
    print(f"Run root: {run_root}")

    out_results = cfg["io"]["write_dir_results"]
    out_logs = cfg["io"]["write_dir_logs"]
    out_ckpt = cfg["train"]["save_dir_checkpoints"]
    ensure_dir(out_results)
    ensure_dir(out_logs)
    ensure_dir(out_ckpt)

    train_video_ids = read_split_txt(str(Path(splits_root) / "train.txt"))
    val_video_ids = read_split_txt(str(Path(splits_root) / "val.txt"))

    video_labels = load_videos_index(videos_index_path)

    label_map = build_canonical_label_map_from_train(train_video_ids, video_labels)
    label_map_path = str(run_label_map_dir / "label_map.json")
    save_label_map(label_map, label_map_path)

    hierarchy = build_hierarchical_maps(label_map)
    class_names = hierarchy["class_names"]
    anomaly_class_names = hierarchy["anomaly_class_names"]
    full_to_anomaly = hierarchy["full_to_anomaly"]
    num_full_classes = int(hierarchy["num_full_classes"])
    num_anomaly_classes = int(hierarchy["num_anomaly_classes"])

    cfg_num_classes = int(cfg["stage2"]["num_classes"])
    if num_full_classes != cfg_num_classes:
        raise ValueError(
            f"label_map size {num_full_classes} != cfg stage2.num_classes {cfg_num_classes}"
        )

    print(f"Full class names: {class_names}")
    print(f"Anomaly-only class names: {anomaly_class_names}")

    train_full_class_ids = [
        label_map[canonical_group_name(video_labels[video_id].group, video_labels[video_id].label_bin)]
        for video_id in train_video_ids
    ]
    val_full_class_ids = [
        label_map[canonical_group_name(video_labels[video_id].group, video_labels[video_id].label_bin)]
        for video_id in val_video_ids
    ]

    train_binary_targets = [0 if class_id == 0 else 1 for class_id in train_full_class_ids]
    anomaly_train_targets = [full_to_anomaly[class_id] for class_id in train_full_class_ids if class_id != 0]

    print("Train full class distribution:", Counter(train_full_class_ids))
    print("Val full class distribution:", Counter(val_full_class_ids))
    print("Train binary distribution:", Counter(train_binary_targets))
    print("Train anomaly-only distribution:", Counter(anomaly_train_targets))

    binary_class_weights = make_class_weights(train_binary_targets, 2).to(device)
    anomaly_class_weights = make_class_weights(anomaly_train_targets, num_anomaly_classes).to(device)

    print("Binary class weights:", [float(x) for x in binary_class_weights.detach().cpu().tolist()])
    print("Anomaly class weights:", [float(x) for x in anomaly_class_weights.detach().cpu().tolist()])

    teacher_train_path = "artifacts/stage2/teacher/teacher_scores_train.pt"
    teacher_val_path = "artifacts/stage2/teacher/teacher_scores_val.pt"
    teacher_scores_train = load_teacher_scores(teacher_train_path)
    teacher_scores_val = load_teacher_scores(teacher_val_path)

    clip_lookup_train = build_clip_lookup_for_split(
        store_root,
        "train",
        expected_dim,
        max_shards_to_scan=int(cfg["io"]["max_shards_to_scan"]),
    )
    clip_lookup_val = build_clip_lookup_for_split(
        store_root,
        "val",
        expected_dim,
        max_shards_to_scan=int(cfg["io"]["max_shards_to_scan"]),
    )

    video_to_clips_train = build_video_to_clips_index(clip_lookup_train, teacher_scores_train)
    video_to_clips_val = build_video_to_clips_index(clip_lookup_val, teacher_scores_val)

    train_dataset = MilVideoDataset(
        train_video_ids,
        video_to_clips_train,
        video_labels,
        embedding_dim=expected_dim,
    )
    val_dataset = MilVideoDataset(
        val_video_ids,
        video_to_clips_val,
        video_labels,
        embedding_dim=expected_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size_videos"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=bool(cfg["train"]["pin_memory"]),
        collate_fn=collate_video_ids,
    )

    sampler_cfg = MgMilConfig(
        bag_size=int(cfg["stage2"]["bag_size"]),
        warmup_epochs=int(cfg["stage2"]["warmup_epochs"]),
        mixture_uniform_ratio=float(cfg["stage2"]["mixture_uniform_ratio"]),
        top_fraction=float(cfg["stage2"]["top_fraction"]),
    )
    sampler = MgMilSampler(cfg=sampler_cfg, seed=seed)

    inject_teacher_score = bool(cfg.get("stage2", {}).get("inject_teacher_score", True))
    input_dim = expected_dim + (1 if inject_teacher_score else 0)

    anomaly_loss_weight = float(cfg.get("train", {}).get("anomaly_loss_weight", 0.5))
    print(f"Teacher score injection: {inject_teacher_score}")
    print(f"Input dimension: {input_dim}")
    print(f"Anomaly loss weight: {anomaly_loss_weight}")

    model = HierarchicalAttentionMilHead(
        embedding_dim=input_dim,
        num_anomaly_classes=num_anomaly_classes,
        attention_hidden_dim=int(cfg["stage2"].get("attention_hidden_dim", 256)),
        dropout=float(cfg["stage2"].get("dropout", 0.1)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    binary_ce = nn.CrossEntropyLoss(weight=binary_class_weights)
    anomaly_ce = FocalLoss(weight=anomaly_class_weights, gamma=2.0)

    epochs = int(cfg["train"]["epochs"])
    grad_clip = float(cfg["train"]["grad_clip_norm"])
    early_patience = int(cfg["train"].get("early_stop_patience", 8))
    early_min_delta = float(cfg["train"].get("early_stop_min_delta", 1e-4))

    best_val_macro_f1 = -1.0
    epochs_without_improve = 0

    log_path = Path(out_results) / "m27_train_log.jsonl"

    for epoch in range(epochs):
        model.train()

        running_total_loss = 0.0
        running_binary_loss = 0.0
        running_anomaly_loss = 0.0
        running_full_loss = 0.0
        running_correct = 0
        running_total = 0

        epoch_start_time = time.time()
        num_batches = 0

        for batch in train_loader:
            num_batches += 1
            batch_video_ids = batch["video_ids"]

            x, mask, y_full, y_binary, y_anomaly = build_bags_for_batch(
                dataset=train_dataset,
                sampler=sampler,
                label_map=label_map,
                epoch=epoch,
                video_ids=batch_video_ids,
                device=device,
                inject_teacher_score=inject_teacher_score,
                full_to_anomaly=full_to_anomaly,
            )

            outputs = model(x, mask)

            binary_loss = binary_ce(outputs.binary_logits, y_binary)

            anomaly_mask = y_binary == 1
            if bool(anomaly_mask.any().item()):
                anomaly_loss = anomaly_ce(outputs.anomaly_logits[anomaly_mask], y_anomaly[anomaly_mask])
            else:
                anomaly_loss = torch.tensor(0.0, device=device)

            full_probs, full_preds = compose_full_preds_from_logits(
                outputs.binary_logits,
                outputs.anomaly_logits,
            )
            full_loss = compute_full_loss_from_probs(full_probs, y_full)

            total_loss = binary_loss + anomaly_loss_weight * anomaly_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            batch_count = int(y_full.shape[0])
            running_total += batch_count
            running_correct += int((full_preds == y_full).sum().item())
            running_total_loss += float(total_loss.item()) * batch_count
            running_binary_loss += float(binary_loss.item()) * batch_count
            running_anomaly_loss += float(anomaly_loss.item()) * batch_count
            running_full_loss += float(full_loss.item()) * batch_count

        train_accuracy = running_correct / max(1, running_total)
        train_loss = running_total_loss / max(1, running_total)
        train_binary_loss = running_binary_loss / max(1, running_total)
        train_anomaly_loss = running_anomaly_loss / max(1, running_total)
        train_full_loss = running_full_loss / max(1, running_total)

        print(
            f"Epoch {epoch}: train batches={num_batches}, videos_seen={running_total}, "
            f"time_sec={time.time() - epoch_start_time:.2f}"
        )

        val_metrics = evaluate(
            model=model,
            dataset=val_dataset,
            sampler=sampler,
            label_map=label_map,
            video_ids=val_video_ids,
            epoch=epoch,
            batch_size_videos=int(cfg["train"]["batch_size_videos"]),
            device=device,
            binary_class_weights=binary_class_weights,
            anomaly_class_weights=anomaly_class_weights,
            num_full_classes=num_full_classes,
            inject_teacher_score=inject_teacher_score,
            full_to_anomaly=full_to_anomaly,
            anomaly_loss_weight=anomaly_loss_weight,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_binary_loss": train_binary_loss,
            "train_anomaly_loss": train_anomaly_loss,
            "train_full_loss": train_full_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_binary_loss": val_metrics["binary_loss"],
            "val_anomaly_loss": val_metrics["anomaly_loss"],
            "val_full_loss": val_metrics["full_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_pred_class_dist": val_metrics["pred_class_dist"],
            "label_map_path": label_map_path,
            "sampler_cfg": sampler_cfg.__dict__,
            "inject_teacher_score": inject_teacher_score,
            "input_dim": input_dim,
            "run_tag": run_tag,
            "class_names": class_names,
            "anomaly_class_names": anomaly_class_names,
            "anomaly_loss_weight": anomaly_loss_weight,
        }

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        print(json.dumps(record, indent=2))

        current_macro_f1 = float(val_metrics["macro_f1"])

        if current_macro_f1 > best_val_macro_f1 + early_min_delta:
            best_val_macro_f1 = current_macro_f1
            epochs_without_improve = 0

            checkpoint_path = Path(out_ckpt) / "mil_best.pt"
            torch.save(
                {
                    "meta": {
                        "stage": "stage2",
                        "artifact": "hier_mil_checkpoint",
                        "epoch": epoch,
                        "best_val_macro_f1": best_val_macro_f1,
                        "expected_dim": expected_dim,
                        "input_dim": input_dim,
                        "num_classes": num_full_classes,
                        "num_anomaly_classes": num_anomaly_classes,
                        "label_map_path": label_map_path,
                        "sampler_cfg": sampler_cfg.__dict__,
                        "inject_teacher_score": inject_teacher_score,
                        "seed": seed,
                        "early_stop_patience": early_patience,
                        "early_stop_min_delta": early_min_delta,
                        "run_tag": run_tag,
                        "model_type": "hierarchical_dual_head_mil",
                        "class_names": class_names,
                        "anomaly_class_names": anomaly_class_names,
                        "anomaly_loss_weight": anomaly_loss_weight,
                    },
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improve += 1
            print(
                f"No improvement in val_macro_f1 for {epochs_without_improve}/{early_patience} epochs "
                f"(best={best_val_macro_f1:.6f}, current={current_macro_f1:.6f})."
            )

        if epochs_without_improve >= early_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best val_macro_f1={best_val_macro_f1:.6f}."
            )
            break

    out_val = Path(out_results) / "m27_val_metrics.json"
    out_val.write_text(
        json.dumps(
            {
                "best_val_macro_f1": best_val_macro_f1,
                "early_stop_patience": early_patience,
                "early_stop_min_delta": early_min_delta,
                "epochs_ran": epoch + 1,
                "run_tag": run_tag,
                "model_type": "hierarchical_dual_head_mil",
            },
            indent=2,
        )
    )
    print(f"Wrote: {out_val}")


if __name__ == "__main__":
    main()