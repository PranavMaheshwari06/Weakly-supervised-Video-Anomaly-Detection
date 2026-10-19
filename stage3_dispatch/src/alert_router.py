"""
Emergency Alert Router — Computer-Aided Dispatch (CAD) simulation.

Reads inference metadata JSON and generates formatted dispatch alerts.
Outputs: alert_log.json + console CAD-style printout.

Usage:
    python -m stage3_mgmil.src.alert_router --metadata artifacts/inference/metadata.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def generate_alert(metadata: Dict) -> Dict:
    """
    From inference metadata, produce a dispatch alert if anomaly detected.
    
    metadata format (from inference_any_video.py):
    {
        "video": "path.mp4",
        "fps": 30,
        "total_frames": 480,
        "binary_anomaly_detected": true,
        "dispatch_services": ["police", "medical"],
        "dispatch_scores": {"police": 0.92, "medical": 0.78, "fire": 0.12},
        "anomaly_intervals": [{"start_frame": 142, "end_frame": 310}],
        "alert_trigger_frame": 142,
    }
    """
    alert = {
        "timestamp": datetime.now().isoformat(),
        "source_video": metadata.get("video", "unknown"),
        "alert_triggered": metadata.get("binary_anomaly_detected", False),
        "dispatch": [],
        "priority": "NONE",
        "anomaly_window": None,
        "confidence": {},
    }

    if not alert["alert_triggered"]:
        alert["priority"] = "CLEAR"
        return alert

    services = metadata.get("dispatch_services", [])
    scores = metadata.get("dispatch_scores", {})
    intervals = metadata.get("anomaly_intervals", [])

    alert["dispatch"] = services
    alert["confidence"] = scores

    # Priority assignment
    if "fire" in services:
        alert["priority"] = "P1-EMERGENCY"
    elif "medical" in services:
        alert["priority"] = "P1-MEDICAL"
    elif "police" in services:
        alert["priority"] = "P2-POLICE"
    else:
        alert["priority"] = "P3-REVIEW"

    if intervals:
        fps = float(metadata["fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("A finite positive source FPS is required")
        first = intervals[0]
        alert["anomaly_window"] = {
            "start_frame": first["start_frame"],
            "end_frame": first["end_frame"],
            "start_sec": round(first["start_frame"] / fps, 2),
            "end_sec": round((first["end_frame"] + 1) / fps, 2),
        }
        trigger_frame = metadata.get("alert_trigger_frame", first["start_frame"])
        alert["alert_latency_sec"] = round(trigger_frame / fps, 2)

    return alert


def print_cad_alert(alert: Dict) -> None:
    """Print formatted CAD-style alert to console."""
    border = "=" * 52
    print(f"\n{border}")
    print(f"  COMPUTER-AIDED DISPATCH (CAD) ALERT")
    print(f"{border}")
    print(f"  Time:       {alert['timestamp']}")
    print(f"  Source:     {alert['source_video']}")
    print(f"  Priority:   {alert['priority']}")
    print(f"  Status:     {'ALERT ACTIVE' if alert['alert_triggered'] else 'ALL CLEAR'}")

    if alert["alert_triggered"]:
        print(f"  Dispatch:   {', '.join(s.upper() for s in alert['dispatch'])}")
        print(f"  Confidence:")
        for svc, score in alert.get("confidence", {}).items():
            bar = "█" * int(score * 20)
            print(f"    {svc:10s}: {bar} {score:.2f}")

        window = alert.get("anomaly_window")
        if window:
            print(f"  Window:     {window['start_sec']}s - {window['end_sec']}s")
            print(f"  Latency:    {alert.get('alert_latency_sec', 'N/A')}s to first detection")

    print(f"{border}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True, help="Path to inference metadata JSON")
    parser.add_argument("--output", type=str, default=None, help="Output alert JSON path")
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text())
    alert = generate_alert(metadata)
    print_cad_alert(alert)

    out_path = args.output or str(Path(args.metadata).parent / "alert_log.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(alert, indent=2))
    print(f"Alert saved: {out_path}")


if __name__ == "__main__":
    main()
