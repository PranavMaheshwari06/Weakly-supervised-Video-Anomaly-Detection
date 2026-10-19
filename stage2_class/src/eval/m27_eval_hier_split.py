from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
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
from stage2_class.src.utils.seed import set_deterministic


def canonical_group_name(group: str, label_bin: int) -> str:
    if int(label_bin) == 0:
        return "Normal"
    return str(group)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
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
    
    
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_checkpoint_path(cfg: Dict, run_tag: str) -> Path:
    explicit = os.environ.get("CKPT_PATH", "").strip()
    if explicit:
        checkpoint_path = Path(explicit)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CKPT_PATH was set but file does not exist: {checkpoint_path}")
        return checkpoint_path

    candidates = [
        Path("artifacts/stage2/runs") / run_tag / "checkpoints" / "mil_best.pt",
        Path("artifacts/stage2/runs") / run_tag / "mil_best.pt",
    ]
    for checkpoint_path in candidates:
        if checkpoint_path.exists():
            return checkpoint_path

    fallback = Path(cfg["train"]["save_dir_checkpoints"]) / "mil_best.pt"
    if not fallback.exists():
        raise FileNotFoundError(
            "Could not resolve checkpoint.\n"
            f"Tried:\n- {candidates[0]}\n- {candidates[1]}\n- {fallback}\n"
            "Fix your run folder, set CKPT_PATH, or update train.save_dir_checkpoints in the cfg."
        )
    print(f"[Warning] Using fallback checkpoint from cfg: {fallback} (run_tag={run_tag})")
    return fallback


def compute_macro_metrics(all_targets: List[int], all_preds: List[int], num_classes: int) -> Dict[str, float]:
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


def compute_confusion_matrix(all_targets: List[int], all_preds: List[int], num_classes: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target_value, pred_value in zip(all_targets, all_preds):
        matrix[target_value][pred_value] += 1
    return matrix


def save_confusion_matrix_png(matrix: List[List[int]], class_names: List[str], out_path: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    arr = np.asarray(matrix, dtype=np.int64)

    plt.figure()
    plt.imshow(arr)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)

    for row_index in range(arr.shape[0]):
        for col_index in range(arr.shape[1]):
            plt.text(col_index, row_index, str(arr[row_index, col_index]), ha="center", va="center")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


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


def build_label_maps_from_train(video_labels: Dict, train_video_ids: List[str]) -> Dict:
    group_set = set()
    for video_id in train_video_ids:
        info = video_labels[video_id]
        group_set.add(canonical_group_name(info.group, info.label_bin))

    class_names = ["Normal"] + sorted([group_name for group_name in group_set if group_name != "Normal"])
    label_map = {group_name: class_id for class_id, group_name in enumerate(class_names)}

    if label_map["Normal"] != 0:
        raise ValueError("Normal must map to class 0")

    anomaly_class_names = class_names[1:]
    full_to_anomaly = {full_class_id: full_class_id - 1 for full_class_id in range(1, len(class_names))}
    anomaly_to_full = {anomaly_index: anomaly_index + 1 for anomaly_index in range(len(anomaly_class_names))}

    return {
        "label_map": label_map,
        "class_names": class_names,
        "anomaly_class_names": anomaly_class_names,
        "full_to_anomaly": full_to_anomaly,
        "anomaly_to_full": anomaly_to_full,
    }


def build_bag_tensor(
    dataset: MilVideoDataset,
    sampler: MgMilSampler,
    label_map: Dict[str, int],
    epoch: int,
    video_id: str,
    device: torch.device,
    inject_teacher_score: bool,
    full_to_anomaly: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clips = dataset.get_video_clips(video_id)
    refs = sampler.select_refs_for_video(clips, epoch=epoch, video_id=video_id)

    bag_size = sampler.cfg.bag_size
    if len(refs) < bag_size:
        while len(refs) < bag_size:
            refs.append(refs[-1])
    refs = refs[:bag_size]

    bag = dataset.build_bag_from_refs(video_id, refs)
    x = bag.embeddings

    if inject_teacher_score:
        teacher_scores = bag.teacher_scores.to(dtype=x.dtype).unsqueeze(1)
        x = torch.cat([x, teacher_scores], dim=1)

    video_info = dataset.video_labels[video_id]
    group_name = canonical_group_name(video_info.group, video_info.label_bin)
    full_target = int(label_map[group_name])
    binary_target = 0 if full_target == 0 else 1
    anomaly_target = -1 if full_target == 0 else int(full_to_anomaly[full_target])

    mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)

    return (
        x.to(device),
        mask,
        torch.tensor(full_target, dtype=torch.int64, device=device),
        torch.tensor(binary_target, dtype=torch.int64, device=device),
        torch.tensor(anomaly_target, dtype=torch.int64, device=device),
    )


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataset: MilVideoDataset,
    sampler: MgMilSampler,
    label_map: Dict[str, int],
    video_ids: List[str],
    epoch: int,
    device: torch.device,
    binary_class_weights: torch.Tensor,
    anomaly_class_weights: torch.Tensor,
    num_classes: int,
    inject_teacher_score: bool,
    full_to_anomaly: Dict[int, int],
    anomaly_loss_weight: float,
) -> Dict:
    model.eval()

    binary_ce = nn.CrossEntropyLoss(weight=binary_class_weights)
    anomaly_ce = FocalLoss(weight=anomaly_class_weights, gamma=2.0)

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_prob_normal: List[float] = []
    all_prob_anomaly: List[float] = []
    all_binary_preds: List[int] = []
    video_ids_used: List[str] = []
    pred_counter: Counter = Counter()

    loss_sum = 0.0
    binary_loss_sum = 0.0
    anomaly_loss_sum = 0.0
    correct = 0

    for video_id in video_ids:
        x, mask, y_full, y_binary, y_anomaly = build_bag_tensor(
            dataset=dataset,
            sampler=sampler,
            label_map=label_map,
            epoch=epoch,
            video_id=video_id,
            device=device,
            inject_teacher_score=inject_teacher_score,
            full_to_anomaly=full_to_anomaly,
        )

        outputs = model(x.unsqueeze(0), mask.unsqueeze(0))

        binary_loss = binary_ce(outputs.binary_logits, y_binary.unsqueeze(0))

        if int(y_binary.item()) == 1:
            anomaly_loss = anomaly_ce(outputs.anomaly_logits, y_anomaly.unsqueeze(0))
        else:
            anomaly_loss = torch.tensor(0.0, device=device)

        total_loss = binary_loss + anomaly_loss_weight * anomaly_loss

        p_bin = torch.softmax(outputs.binary_logits, dim=1)
        full_probs = HierarchicalAttentionMilHead.compose_full_probs(
            outputs.binary_logits,
            outputs.anomaly_logits,
        )

        pred = int(torch.argmax(full_probs, dim=1).item())
        target = int(y_full.item())
        binary_pred = int(torch.argmax(p_bin, dim=1).item())

        prob_normal = float(full_probs[0, 0].item())
        prob_anomaly = float(p_bin[0, 1].item())

        all_preds.append(pred)
        all_targets.append(target)
        all_prob_normal.append(prob_normal)
        all_prob_anomaly.append(prob_anomaly)
        all_binary_preds.append(binary_pred)
        video_ids_used.append(video_id)
        pred_counter.update([pred])

        loss_sum += float(total_loss.item())
        binary_loss_sum += float(binary_loss.item())
        anomaly_loss_sum += float(anomaly_loss.item())
        correct += 1 if pred == target else 0

    macro = compute_macro_metrics(all_targets, all_preds, num_classes)
    confusion = compute_confusion_matrix(all_targets, all_preds, num_classes)

    return {
        "count_videos": len(video_ids),
        "accuracy": correct / max(1, len(video_ids)),
        "avg_loss": loss_sum / max(1, len(video_ids)),
        "avg_binary_loss": binary_loss_sum / max(1, len(video_ids)),
        "avg_anomaly_loss": anomaly_loss_sum / max(1, len(video_ids)),
        "macro_recall": macro["macro_recall"],
        "macro_f1": macro["macro_f1"],
        "pred_class_dist": dict(pred_counter),
        "confusion_matrix": confusion,
        "per_video": {
            "video_ids": video_ids_used,
            "targets": all_targets,
            "preds": all_preds,
            "binary_preds": all_binary_preds,
            "prob_normal": all_prob_normal,
            "prob_anomaly": all_prob_anomaly,
        },
    }


def compute_binary_metrics(y_true_bin: List[int], y_score: List[float], threshold: float) -> Dict[str, float]:
    y_true = torch.tensor(y_true_bin, dtype=torch.int64)
    scores = torch.tensor(y_score, dtype=torch.float32)
    y_pred = (scores >= float(threshold)).to(torch.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def choose_threshold_best_f1_binary(y_true_bin: List[int], y_score: List[float]) -> Dict[str, float]:
    scores = torch.tensor(y_score, dtype=torch.float32)
    unique_thresholds = torch.unique(scores).sort().values

    best = {
        "threshold": float(unique_thresholds[0].item()),
        "precision": 0.0,
        "recall": 0.0,
        "f1": -1.0,
    }

    for threshold_tensor in unique_thresholds:
        threshold_value = float(threshold_tensor.item())
        pack = compute_binary_metrics(y_true_bin, y_score, threshold=threshold_value)

        if (pack["f1"] > best["f1"]) or (
            abs(pack["f1"] - best["f1"]) < 1e-12 and pack["recall"] > best["recall"]
        ) or (
            abs(pack["f1"] - best["f1"]) < 1e-12
            and abs(pack["recall"] - best["recall"]) < 1e-12
            and threshold_value < best["threshold"]
        ):
            best = {"threshold": threshold_value, **pack}

    return best


def compute_auroc_binary(y_true_bin: List[int], y_score: List[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true_bin, y_score))
    except Exception:
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=os.environ.get("STAGE2_CFG", "stage2_class/configs/hierA_dual_head.yaml"))
    parser.add_argument("--split", type=str, default=os.environ.get("SPLIT", "test"))
    parser.add_argument("--run_tag", type=str, default=os.environ.get("RUN_TAG", "hier_stage2_run"))
    args = parser.parse_args()

    cfg_path = args.cfg
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    seed = int(cfg.get("seed", 1337))
    set_deterministic(seed)

    split_name = args.split.strip().lower()
    if split_name not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")

    run_tag = args.run_tag.strip()

    store_root = cfg["paths"]["feature_store_root"]
    splits_root = cfg["paths"]["splits_root"]
    videos_index_path = cfg["paths"]["videos_index_jsonl"]
    expected_dim = int(cfg["stage1"]["global_dim"])
    num_classes = int(cfg["stage2"]["num_classes"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    )
    print(f"Device: {device} | split={split_name} | run_tag={run_tag}")

    train_video_ids = read_split_txt(str(Path(splits_root) / "train.txt"))
    split_video_ids = read_split_txt(str(Path(splits_root) / f"{split_name}.txt"))

    video_labels = load_videos_index(videos_index_path)

    mapping_pack = build_label_maps_from_train(video_labels, train_video_ids)
    label_map = mapping_pack["label_map"]
    class_names = mapping_pack["class_names"]
    anomaly_class_names = mapping_pack["anomaly_class_names"]
    full_to_anomaly = mapping_pack["full_to_anomaly"]

    if len(label_map) != num_classes:
        raise ValueError(f"Label map size {len(label_map)} != num_classes {num_classes}")

    train_full_targets: List[int] = []
    for video_id in train_video_ids:
        info = video_labels[video_id]
        group_name = canonical_group_name(info.group, info.label_bin)
        train_full_targets.append(label_map[group_name])

    train_binary_targets = [0 if target == 0 else 1 for target in train_full_targets]
    train_anomaly_targets = [full_to_anomaly[target] for target in train_full_targets if target != 0]

    binary_class_weights = make_class_weights(train_binary_targets, 2).to(device)
    anomaly_class_weights = make_class_weights(train_anomaly_targets, len(anomaly_class_names)).to(device)

    teacher_path = f"artifacts/stage2/teacher/teacher_scores_{split_name}.pt"
    teacher_scores = load_teacher_scores(teacher_path)

    clip_lookup = build_clip_lookup_for_split(
        store_root,
        split_name,
        expected_dim,
        max_shards_to_scan=int(cfg["io"]["max_shards_to_scan"]),
    )
    video_to_clips = build_video_to_clips_index(clip_lookup, teacher_scores)
    dataset = MilVideoDataset(split_video_ids, video_to_clips, video_labels, embedding_dim=expected_dim)

    sampler_cfg = MgMilConfig(
        bag_size=int(cfg["stage2"]["bag_size"]),
        warmup_epochs=int(cfg["stage2"]["warmup_epochs"]),
        mixture_uniform_ratio=float(cfg["stage2"]["mixture_uniform_ratio"]),
        top_fraction=float(cfg["stage2"]["top_fraction"]),
    )
    sampler = MgMilSampler(cfg=sampler_cfg, seed=seed)

    checkpoint_path = resolve_checkpoint_path(cfg=cfg, run_tag=run_tag)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    inject_teacher_score = bool(checkpoint["meta"].get("inject_teacher_score", True))
    input_dim = int(checkpoint["meta"].get("input_dim", expected_dim + 1))
    anomaly_loss_weight = float(checkpoint["meta"].get("anomaly_loss_weight", 1.0))

    model = HierarchicalAttentionMilHead(
        embedding_dim=input_dim,
        num_anomaly_classes=len(anomaly_class_names),
        attention_hidden_dim=int(cfg["stage2"].get("attention_hidden_dim", 256)),
        dropout=float(cfg["stage2"].get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    metrics = evaluate_split(
        model=model,
        dataset=dataset,
        sampler=sampler,
        label_map=label_map,
        video_ids=split_video_ids,
        epoch=0,
        device=device,
        binary_class_weights=binary_class_weights,
        anomaly_class_weights=anomaly_class_weights,
        num_classes=num_classes,
        inject_teacher_score=inject_teacher_score,
        full_to_anomaly=full_to_anomaly,
        anomaly_loss_weight=anomaly_loss_weight,
    )

    out_dir = Path("artifacts/stage2/eval") / run_tag
    ensure_dir(out_dir)

    phase2_path = out_dir / f"phase2_{split_name}.json"
    phase2_payload = {
        "meta": {
            "split": split_name,
            "run_tag": run_tag,
            "cfg_path": str(cfg_path),
            "checkpoint": str(checkpoint_path.resolve()),
            "best_val_macro_f1": float(checkpoint["meta"].get("best_val_macro_f1", -1.0)),
            "trained_epoch": int(checkpoint["meta"].get("epoch", -1)),
            "sampler_cfg": checkpoint["meta"].get("sampler_cfg", {}),
            "inject_teacher_score": inject_teacher_score,
            "input_dim": input_dim,
            "seed": seed,
            "class_names": class_names,
            "anomaly_class_names": anomaly_class_names,
            "normal_class_index": 0,
            "model_type": checkpoint["meta"].get("model_type", "hierarchical_dual_head_mil"),
            "anomaly_loss_weight": anomaly_loss_weight,
        },
        "metrics": {
            "count_videos": metrics["count_videos"],
            "accuracy": metrics["accuracy"],
            "avg_loss": metrics["avg_loss"],
            "avg_binary_loss": metrics["avg_binary_loss"],
            "avg_anomaly_loss": metrics["avg_anomaly_loss"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "pred_class_dist": metrics["pred_class_dist"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "per_video": metrics["per_video"],
    }
    phase2_path.write_text(json.dumps(phase2_payload, indent=2))
    print(f"Wrote: {phase2_path}")

    prob_anomaly = metrics["per_video"]["prob_anomaly"]
    y_score = [float(value) for value in prob_anomaly]

    y_true_bin: List[int] = []
    for video_id in metrics["per_video"]["video_ids"]:
        y_true_bin.append(int(video_labels[video_id].label_bin))

    threshold_path = out_dir / "phase2_binary_threshold.json"
    if split_name == "val":
        threshold_pack = choose_threshold_best_f1_binary(y_true_bin, y_score)
        threshold_payload = {
            "threshold": float(threshold_pack["threshold"]),
            "selected_on": "val",
            "policy": "best_f1",
            "score_definition": "P(Binary=Anomaly)",
            "normal_class_index": 0,
        }
        threshold_path.write_text(json.dumps(threshold_payload, indent=2))
        threshold = float(threshold_pack["threshold"])
    else:
        if not threshold_path.exists():
            raise FileNotFoundError(f"Missing {threshold_path}. Run split=val first to select binary threshold.")
        threshold = float(json.loads(threshold_path.read_text())["threshold"])

    binary_pack = compute_binary_metrics(y_true_bin, y_score, threshold=threshold)
    binary_pack["auroc"] = compute_auroc_binary(y_true_bin, y_score)
    binary_pack["threshold"] = float(threshold)
    binary_pack["score_definition"] = "P(Binary=Anomaly)"
    binary_pack["normal_class_index"] = 0
    binary_pack["count_videos"] = int(len(y_true_bin))

    binary_path = out_dir / f"phase2_binary_{split_name}.json"
    binary_path.write_text(json.dumps(binary_pack, indent=2))
    print(f"Wrote: {binary_path}")

    if split_name == "test":
        confusion_json_path = out_dir / "phase2_confusion_test.json"
        confusion_png_path = out_dir / "phase2_confusion_test.png"
        confusion_json_path.write_text(
            json.dumps({"class_names": class_names, "matrix": metrics["confusion_matrix"]}, indent=2)
        )
        save_confusion_matrix_png(metrics["confusion_matrix"], class_names, confusion_png_path)
        print(f"Wrote: {confusion_json_path}")
        print(f"Wrote: {confusion_png_path}")


if __name__ == "__main__":
    main()