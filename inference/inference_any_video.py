"""
Inference on any UCF-Crime video — supports 14-class and dispatch modes.
Outputs annotated video, metadata JSON, and CAD-style dispatch alert.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import yaml
from stage2_class.src.eval.temporal_geometry import (
    attention_intervals, clip_support, smooth_attention, validate_metadata,
)

from stage2_class.src.data.mil_index import (
    build_clip_lookup_for_split,
    build_video_to_clips_index,
    load_teacher_scores,
)
from stage2_class.src.models.mil_hierarchical import HierarchicalAttentionMilHead
from stage2_class.src.utils.dispatch_label_map import (
    DISPATCH_CHANNELS,
    NUM_DISPATCH,
    dispatch_vector_to_services,
)


def find_video_in_splits(target: str, store_root: str, expected_dim: int):
    """Search train/val/test splits for the target video features."""
    for split in ["test", "val", "train"]:
        teacher = load_teacher_scores(f"artifacts/stage2/teacher/teacher_scores_{split}.pt")
        clip_lookup = build_clip_lookup_for_split(store_root, split, expected_dim, -1)
        vtc = build_video_to_clips_index(clip_lookup, teacher)

        for vid_path in vtc.keys():
            if target in vid_path:
                return vtc[vid_path], vid_path
    return None, None


def extract_intervals(attn, clip_starts, metadata, threshold: float = 0.65):
    """Convert half-open source support to the renderer's inclusive end schema."""
    return [{"start_frame": start, "end_frame": end - 1}
            for start, end in attention_intervals(attn, clip_starts, metadata, threshold)]


def run_inference(args):
    cfg = yaml.safe_load(Path("stage2_class/configs/hierA_dispatch_v1.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store_root = cfg["paths"]["feature_store_root"]
    expected_dim = int(cfg["stage1"]["global_dim"])

    # Find video features
    clips_list, vid_path = find_video_in_splits(args.video_id, store_root, expected_dim)
    if clips_list is None:
        print(f"ERROR: Could not find features for '{args.video_id}'")
        return

    raw_video = f"data/ucf_crime/raw/{vid_path}"
    print(f"Found: {vid_path} | Raw: {raw_video}")

    ckpt_path = f"artifacts/stage2/runs/{args.run_tag}/checkpoints/mil_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    inject_teacher = bool(ckpt["meta"].get("inject_teacher_score", True))
    input_dim = int(ckpt["meta"].get("input_dim", expected_dim + 1))

    if args.mode == "dispatch":
        num_out = NUM_DISPATCH
    else:
        num_out = int(ckpt["meta"].get("num_anomaly_classes", 13))

    model = HierarchicalAttentionMilHead(
        embedding_dim=input_dim,
        num_anomaly_classes=num_out,
        attention_hidden_dim=256,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # use all clips for temporal coverage (no subsampling)
    clips = sorted(clips_list, key=lambda c: c.clip_start)
    x_list, starts = [], []
    for c in clips:
        shard = torch.load(c.shard_path, map_location="cpu")
        idx = shard["clip_keys"].index(c.clip_key)
        emb = shard["global"][idx].float()
        if inject_teacher:
            score = torch.tensor([c.teacher_score], dtype=torch.float32)
            emb = torch.cat([emb, score])
        x_list.append(emb)
        starts.append(c.clip_start)

    x = torch.stack(x_list).unsqueeze(0).to(device)
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model(x, mask)
        attn = out.attention.squeeze(0).cpu().numpy()


        p_bin = torch.softmax(out.binary_logits, dim=1)
        is_anomaly = bool(torch.argmax(p_bin, dim=1).item() == 1)
        anomaly_conf = float(p_bin[0, 1].item())


        if args.mode == "dispatch":
            disp_scores_raw = torch.sigmoid(out.anomaly_logits).squeeze(0).cpu().tolist()
            dispatch_scores = {DISPATCH_CHANNELS[i]: round(disp_scores_raw[i], 4) for i in range(NUM_DISPATCH)}
            # per-channel thresholds for precision-recall balance
            disp_thresholds = [0.65, 0.60, 0.65]  # police, medical, fire
            services = [DISPATCH_CHANNELS[i] for i in range(NUM_DISPATCH) if disp_scores_raw[i] >= disp_thresholds[i]]
            crime_label = " + ".join(s.upper() for s in services) if services else "NORMAL"
        else:
            label_map_path = f"artifacts/stage2/runs/{args.run_tag}/label_map/label_map.json"
            label_map = json.loads(Path(label_map_path).read_text())
            anomaly_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1]) if k != "Normal"]
            pred_idx = int(torch.argmax(out.anomaly_logits, dim=1).item())
            crime_label = anomaly_names[pred_idx] if pred_idx < len(anomaly_names) else "UNKNOWN"
            dispatch_scores = {}
            services = []

    # Use the same integer decimation metadata as feature extraction.
    records = [json.loads(line) for line in Path(cfg["paths"]["videos_index_jsonl"]).read_text().splitlines() if line.strip()]
    matches = [record for record in records if record["rel_path"] == vid_path]
    if len(matches) != 1:
        raise ValueError(f"Expected one metadata record: {vid_path}")
    temporal_meta = matches[0]
    validate_metadata(temporal_meta)
    cap = cv2.VideoCapture(raw_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open source video: {raw_video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if total_frames != temporal_meta["src_frame_count"] or not np.isclose(fps, temporal_meta["src_fps"], rtol=0, atol=1e-6):
        raise ValueError("Source metadata differs from the feature index")
    intervals = extract_intervals(attn, starts, temporal_meta)
    trigger_frame = intervals[0]["start_frame"] if intervals else 0
    smoothed = smooth_attention(attn, starts, temporal_meta)
    clip_metadata = []
    for i, (clip_start, score) in enumerate(zip(starts, smoothed)):
        start_frame, end_frame = clip_support(clip_start, temporal_meta)
        clip_metadata.append({
            "clip_idx": i,
            "frame_start": start_frame,
            "frame_end": end_frame - 1,
            "anomaly_score": round(float(score), 4),
        })


    metadata = {
        "video": raw_video,
        "video_id": vid_path,
        "fps": fps,
        "total_frames": total_frames,
        "mode": args.mode,
        "binary_anomaly_detected": is_anomaly,
        "anomaly_confidence": round(anomaly_conf, 4),
        "crime_label": crime_label,
        "dispatch_services": services,
        "dispatch_scores": dispatch_scores,
        "anomaly_intervals": intervals,
        "alert_trigger_frame": trigger_frame,
        "clips": clip_metadata,
    }

    out_dir = Path("artifacts/inference") / args.run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    vid_stem = Path(vid_path).stem
    meta_path = out_dir / f"{vid_stem}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata: {meta_path}")


    from stage3_dispatch.src.alert_router import generate_alert, print_cad_alert
    alert = generate_alert(metadata)
    print_cad_alert(alert)
    alert_path = out_dir / f"{vid_stem}_alert.json"
    alert_path.write_text(json.dumps(alert, indent=2))


    from stage3_dispatch.src.visualize_dispatch_video import render_video
    video_out = Path("artifacts/visualizations") / args.run_tag / f"{vid_stem}_{args.mode}.mp4"
    render_video(metadata, raw_video, str(video_out))
    print(f"Video: {video_out}")


def main():
    parser = argparse.ArgumentParser(description="Inference on any UCF-Crime video")
    parser.add_argument("--video_id", type=str, required=True, help="Video identifier (e.g. Shooting/Shooting001_x264)")
    parser.add_argument("--run_tag", type=str, default="dispatch_v1", help="Model run tag")
    parser.add_argument("--mode", type=str, choices=["dispatch", "hierarchical"], default="dispatch")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
