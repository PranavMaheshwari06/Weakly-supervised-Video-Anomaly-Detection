""" Stage 1b Feature Extraction: Re-extract global features using LoRA-adapted VideoMAE.
Produces shard files in EXACTLY the same format as Stage-1 frozen features, so the
entire Stage-2 MIL pipeline works unchanged — just point it at the new feature directory.

Output shard schema (identical to existing):
{
  "meta": {...},
  "clip_keys": List[str],
  "video_ids": List[str],
  "clip_starts": int32[N],
  "labels": int64[N],
  "global": float16[N, 768]
}

Usage:
python -m stage1b_finetune.src.extract_lora_features \
  --cfg stage1b_finetune/configs/lora_finetune.yaml \
  --lora_ckpt artifacts/stage1b/checkpoints/lora_best.pt \
  --split train

Hardware: RTX 4050 (6 GB) — batch_size=4 clips at inference is safe.
"""

from __future__ import annotations
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import yaml
from torch.amp import autocast

from stage1b_finetune.src.lora_backbone import LoRAVideoMAE
from stage1b_finetune.src.clip_dataset import (
    decode_clip_frames,
    load_videos_index,
    make_clip_key,
    read_split_txt,
    transform_clip,
    VideoRecord,
)


def deterministic_sample_starts(
    all_starts: List[int],
    video_id: str,
    max_clips: int,
    seed: int,
) -> List[int]:
    """Deterministic clip sampling matching Stage-1 behavior."""
    import random
    starts_sorted = sorted(all_starts)
    if len(starts_sorted) <= max_clips:
        return starts_sorted
    
    video_hash = abs(hash(video_id)) % (10**9)
    rng = random.Random(seed + video_hash)
    chosen = rng.sample(starts_sorted, k=max_clips)
    return sorted(chosen)


def atomic_save_pt(payload: Dict, out_path: str) -> None:
    """Atomic save to prevent corruption."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pt", dir=os.path.dirname(out_path))
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def extract_split(
    model: LoRAVideoMAE,
    videos_index: Dict[str, VideoRecord],
    video_ids: List[str],
    raw_root: str,
    out_dir: str,
    split_name: str,
    max_clips_per_video: int,
    batch_size_clips: int,
    shard_size: int,
    seed: int,
    device: torch.device,
) -> List[str]:
    """
    Extract features for all videos in a split, outputting shards.
    """
    os.makedirs(out_dir, exist_ok=True)
    shard_paths: List[str] = []
    shard_id = 0

    # Buffers for current shard
    buf_clip_keys: List[str] = []
    buf_video_ids: List[str] = []
    buf_clip_starts: List[int] = []
    buf_labels: List[int] = []
    buf_features: List[torch.Tensor] = []

    def flush_shard() -> None:
        nonlocal shard_id, buf_clip_keys, buf_video_ids, buf_clip_starts, buf_labels, buf_features
        if not buf_clip_keys:
            return

        global_mat = torch.cat(buf_features, dim=0).to(torch.float16)

        meta = {
            "stage": "stage1_features",
            "version": "dman_v21",
            "backbone": "MCG-NJU/videomae-base+LoRA",
            "fps_target": 15,
            "clip_len": 16,
            "stride": 8,
            "global_pool": "mean(tokens)",
            "dtype_disk": "float16",
            "split": split_name,
            "lora_adapted": True,
        }

        payload = {
            "meta": meta,
            "clip_keys": buf_clip_keys,
            "video_ids": buf_video_ids,
            "clip_starts": torch.tensor(buf_clip_starts, dtype=torch.int32),
            "labels": torch.tensor(buf_labels, dtype=torch.int64),
            "global": global_mat,
        }

        out_path = os.path.join(out_dir, f"global_{split_name}_shard_{shard_id:03d}.pt")
        atomic_save_pt(payload, out_path)
        shard_paths.append(out_path)
        shard_id += 1

        buf_clip_keys = []
        buf_video_ids = []
        buf_clip_starts = []
        buf_labels = []
        buf_features = []

    print(f"[extract] split={split_name} | videos={len(video_ids)} | max_clips={max_clips_per_video}")

    for v_i, video_id in enumerate(video_ids, start=1):
        if video_id not in videos_index:
            continue
        rec = videos_index[video_id]
        video_path = str(Path(raw_root) / rec.rel_path)
        if not Path(video_path).exists():
            print(f"[extract] SKIP missing: {video_id}")
            continue

        # Select clip starts
        starts = deterministic_sample_starts(
            rec.clip_starts, video_id, max_clips_per_video, seed
        )
        if not starts:
            continue

        # Decode all clips for this video
        clip_tensors: List[torch.Tensor] = []
        valid_starts: List[int] = []

        for cs in starts:
            frames = decode_clip_frames(
                video_path=video_path,
                clip_start_15fps=cs,
                clip_len=16,
                keep_every_n=rec.keep_every_n,
            )
            if frames is None:
                continue
            clip_tensor = transform_clip(frames, img_size=224)
            clip_tensors.append(clip_tensor)
            valid_starts.append(cs)

        if not clip_tensors:
            continue

        # Forward through model in batches
        all_features: List[torch.Tensor] = []
        for i in range(0, len(clip_tensors), batch_size_clips):
            batch = torch.stack(clip_tensors[i:i + batch_size_clips]).to(device)
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float16):
                features = model.extract_features(batch)  # [B, 768]
                all_features.append(features.cpu().float())

        features_cat = torch.cat(all_features, dim=0)  # [N_clips, 768]

        # Add to shard buffer
        for j, cs in enumerate(valid_starts):
            buf_clip_keys.append(make_clip_key(video_id, cs))
            buf_video_ids.append(video_id)
            buf_clip_starts.append(cs)
            buf_labels.append(int(rec.label_bin))
            buf_features.append(features_cat[j:j+1])

        # Flush shard if buffer is full
        if len(buf_clip_keys) >= shard_size:
            print(f"[extract] flush shard {shard_id:03d} | clips={len(buf_clip_keys)}")
            flush_shard()

        # Progress
        if v_i % 25 == 0:
            print(f"[extract] {v_i}/{len(video_ids)} videos | clips_buffered={len(buf_clip_keys)}")

    # Final flush
    if buf_clip_keys:
        print(f"[extract] final flush shard {shard_id:03d} | clips={len(buf_clip_keys)}")
        flush_shard()

    return shard_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features with LoRA backbone")
    parser.add_argument("--cfg", type=str, default="stage1b_finetune/configs/lora_finetune.yaml")
    parser.add_argument("--lora_ckpt", type=str, default="artifacts/stage1b/checkpoints/lora_best.pt")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"LoRA checkpoint: {args.lora_ckpt}")
    print(f"Split: {args.split}")

    # Load model
    print("Loading LoRA model...")
    model = LoRAVideoMAE.load_from_checkpoint(
        checkpoint_path=args.lora_ckpt,
        device=str(device),
        use_gradient_checkpointing=False,  # Not needed for inference
    )
    model.eval()

    # Load data index
    videos_index = load_videos_index(cfg["paths"]["videos_jsonl"])
    split_video_ids = read_split_txt(str(Path(cfg["paths"]["splits_root"]) / f"{args.split}.txt"))
    split_video_ids = [v for v in split_video_ids if v in videos_index]
    print(f"Videos to process: {len(split_video_ids)}")

    # Determine max clips
    extract_cfg = cfg["extract"]
    if args.split == "train":
        max_clips = int(extract_cfg["max_clips_per_video"])
    else:
        max_clips = int(extract_cfg["max_clips_per_video_eval"])

    # Extract
    t0 = time.time()
    shard_paths = extract_split(
        model=model,
        videos_index=videos_index,
        video_ids=split_video_ids,
        raw_root=cfg["paths"]["raw_video_root"],
        out_dir=cfg["paths"]["lora_feature_store"],
        split_name=args.split,
        max_clips_per_video=max_clips,
        batch_size_clips=int(extract_cfg["batch_size_clips"]),
        shard_size=int(extract_cfg["shard_size"]),
        seed=int(cfg["seed"]),
        device=device,
    )
    elapsed = time.time() - t0

    print(f"\nDone! Extracted {args.split} in {elapsed:.1f}s")
    print(f"Shards written: {len(shard_paths)}")
    for p in shard_paths[:5]:
        print(f"  {p}")
    if len(shard_paths) > 5:
        print(f"  ... ({len(shard_paths)} total)")


if __name__ == "__main__":
    main()