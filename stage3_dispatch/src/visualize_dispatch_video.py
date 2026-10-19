"""
Dispatch Video Visualizer.

Reads inference metadata and raw video, produces annotated output with:
- Top banner: dispatch services + confidence
- Color-coded border (green=normal, red=anomaly)
- Bottom heatmap bar: anomaly score per clip
- Dispatch icons overlay

Usage:
    python -m stage3_dispatch.src.visualize_dispatch_video \
        --metadata artifacts/inference/metadata.json \
        --video data/ucf_crime/raw/Shooting/Shooting001_x264.mp4 \
        --output artifacts/visualizations/Shooting001_dispatch.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# Service colors (BGR)
SERVICE_COLORS = {
    "police": (255, 100, 0),    # Blue
    "medical": (0, 0, 255),     # Red
    "fire": (0, 165, 255),      # Orange
}

SERVICE_ICONS = {
    "police": "[POLICE]",
    "medical": "[MEDICAL]",
    "fire": "[FIRE]",
}


def build_frame_score_map(metadata: Dict) -> np.ndarray:
    """Build per-frame anomaly score array from clip-level data."""
    total_frames = metadata.get("total_frames", 0)
    if total_frames == 0:
        return np.array([])

    scores = np.zeros(total_frames, dtype=np.float32)
    clips = metadata.get("clips", [])

    for clip in clips:
        s = clip["frame_start"]
        e = min(clip["frame_end"], total_frames - 1)
        scores[s:e + 1] = clip["anomaly_score"]

    return scores


def is_in_anomaly(frame_idx: int, intervals: List[Dict]) -> bool:
    for iv in intervals:
        if iv["start_frame"] <= frame_idx <= iv["end_frame"]:
            return True
    return False


def draw_heatmap_bar(frame: np.ndarray, score: float, y_offset: int, bar_height: int = 8) -> None:
    """Draw a thin heatmap bar at bottom showing current anomaly intensity."""
    h, w = frame.shape[:2]
    # Score to color: green (0) → red (1)
    r = int(255 * score)
    g = int(255 * (1 - score))
    cv2.rectangle(frame, (0, y_offset), (w, y_offset + bar_height), (0, g, r), -1)


def render_video(metadata: Dict, video_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # upscale for HD text rendering
    # UCF-Crime videos are often 320x240. Text looks terrible at this resolution.
    # We upscale the frame to at least 720p to render crisp, HD text overlays.
    min_h = 720
    scale_factor = max(1.0, min_h / float(h))
    out_w = int(w * scale_factor)
    out_h = int(h * scale_factor)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Try mp4v first as it is more widely supported on Linux/WSL than avc1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print("Warning: mp4v failed, falling back to XVID (.avi)")
        output_path = output_path.replace(".mp4", ".avi")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (out_w, out_h))

    frame_scores = build_frame_score_map(metadata)
    
    # Only show anomalies and dispatch if the overall video was classified as anomalous!
    is_video_anomalous = metadata.get("binary_anomaly_detected", False)
    intervals = metadata.get("anomaly_intervals", []) if is_video_anomalous else []
    services = metadata.get("dispatch_services", []) if is_video_anomalous else []
    dispatch_scores = metadata.get("dispatch_scores", {}) if is_video_anomalous else {}

    # Calculate scales based on the HD output resolution
    base_scale = out_w / 800.0
    font_scale = max(0.5, base_scale * 0.9)
    thick = max(1, int(2.5 * base_scale))
    font = cv2.FONT_HERSHEY_DUPLEX
    
    banner_h = int(75 * max(0.6, base_scale))
    bar_h = int(10 * max(1, base_scale))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Upscale frame for HD text rendering
        if scale_factor > 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        in_anomaly = is_in_anomaly(frame_idx, intervals)
        score = float(frame_scores[frame_idx]) if frame_idx < len(frame_scores) else 0.0

        # draw border
        border_color = (0, 0, 255) if in_anomaly else (0, 200, 0)
        cv2.rectangle(frame, (0, 0), (out_w - 1, out_h - 1), border_color, thick + 1)

        # top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (out_w, banner_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # 1. Time counter (Top Left)
        txt_y = int(banner_h * 0.40)
        current_sec = int(frame_idx / fps) if fps > 0 else 0
        total_sec = int(total / fps) if fps > 0 else 0
        current_str = f"{current_sec // 60:02d}:{current_sec % 60:02d}"
        total_str = f"{total_sec // 60:02d}:{total_sec % 60:02d}"
        cv2.putText(frame, f"Time: {current_str} / {total_str}", (8, txt_y),
                    font, font_scale * 0.7, (220, 220, 220), thick)

        # 2. Status (Top Right): one color-coded tag per dispatched service
        if in_anomaly and services:
            # White keeps the "ALERT:" prefix distinct from Medical's red service tag.
            parts = [("ALERT:", (255, 255, 255))] + [(SERVICE_ICONS[s], SERVICE_COLORS[s]) for s in services]
        elif in_anomaly:
            parts = [("ANOMALY DETECTED", (0, 100, 255))]
        else:
            parts = [("STATUS: NORMAL", (0, 255, 0))]

        widths = [cv2.getTextSize(t, font, font_scale, thick)[0][0] for t, _ in parts]
        gap = 8
        x = out_w - 8 - sum(widths) - gap * (len(parts) - 1)
        for (t, c), tw in zip(parts, widths):
            cv2.putText(frame, t, (x, txt_y), font, font_scale, c, thick)
            x += tw + gap

        # 3. Confidence scores (Second line, bottom left), dispatched services in their color
        txt_y2 = int(banner_h * 0.85)
        conf_scale = font_scale * 0.65
        conf_thick = max(1, thick - 1)
        x = 8
        for svc in ["police", "medical", "fire"]:
            if is_video_anomalous and in_anomaly and svc in dispatch_scores:
                val = dispatch_scores[svc]
            else:
                val = 0.0
            txt = f"{svc}: {val:.0%}   "
            c = SERVICE_COLORS[svc] if in_anomaly and svc in services else (120, 120, 120)
            cv2.putText(frame, txt, (x, txt_y2), font, conf_scale, c, conf_thick)
            x += cv2.getTextSize(txt, font, conf_scale, conf_thick)[0][0]

        # Frame counter (second line, right; the top line is full when three services alert)
        frame_txt = f"Frame: {frame_idx} / {total}"
        sz = cv2.getTextSize(frame_txt, font, conf_scale, conf_thick)[0]
        cv2.putText(frame, frame_txt, (out_w - sz[0] - 8, txt_y2),
                    font, conf_scale, (220, 220, 220), conf_thick)

        # bottom heatmap bar
        draw_heatmap_bar(frame, score, out_h - bar_h - 2, bar_h)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Rendered {frame_idx} frames → {output_path}")



    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text())

    output = args.output
    if not output:
        stem = Path(args.video).stem
        output = f"artifacts/visualizations/{stem}_dispatch.mp4"

    render_video(metadata, args.video, output)


if __name__ == "__main__":
    main()  