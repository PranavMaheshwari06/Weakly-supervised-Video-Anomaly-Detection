from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional, Set

import cv2
import torch
import torchvision.transforms.functional as F

from stage1_features.src.videomae_backbone import load_videomae, extract_batch



def utc_iso() -> str:
    # timezone-aware UTC (avoids deprecated utcnow warning)
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit_or_unknown(repo_root: str) -> str:
    # If repo isn't a git repo, return "unknown" instead of printing fatal errors.
    try:
        out = subprocess.check_output(["git", "-C", repo_root, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


def make_clip_key(video_id: str, clip_start: int) -> str:
    # Locked deterministic key format
    return f"{video_id}__s{clip_start:06d}"


def read_jsonl(path_jsonl: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def read_split_txt(path_txt: str) -> List[str]:
    with open(path_txt, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def deterministic_sample(starts: List[int], video_id: str, max_k: int, seed: int) -> List[int]:
    starts_sorted = sorted(int(x) for x in starts)
    if max_k <= 0:
        return []
    if len(starts_sorted) <= max_k:
        return starts_sorted

    # stable per-video sampling
    video_hash = abs(hash(video_id)) % (10**9)
    rng = random.Random(seed + video_hash)
    chosen = rng.sample(starts_sorted, k=max_k)
    return sorted(chosen)


def atomic_save_pt(payload: Dict[str, Any], out_path: str) -> None:
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


def _resize_center_crop_224(rgb_uint8_hwc: torch.Tensor) -> torch.Tensor:
    """
    rgb_uint8_hwc: torch.uint8 (H,W,3) RGB
    returns: torch.float32 (3,224,224) in [0,1]
    """
    if rgb_uint8_hwc.dtype != torch.uint8 or rgb_uint8_hwc.ndim != 3 or rgb_uint8_hwc.shape[-1] != 3:
        raise ValueError(f"Expected uint8 (H,W,3), got {rgb_uint8_hwc.dtype} {tuple(rgb_uint8_hwc.shape)}")

    img = rgb_uint8_hwc.permute(2, 0, 1).contiguous().float() / 255.0  # (3,H,W)
    img_resized = F.resize(img, size=224, antialias=True)              # shortest side -> 224
    img_cropped = F.center_crop(img_resized, output_size=[224, 224])
    return img_cropped



def decode_needed_frames_sequential(
    video_path: str,
    start_orig: int,
    end_orig: int,
    needed_orig_indices: Set[int],
) -> Dict[int, torch.Tensor]:
    """
    Seek once to start_orig, then read frames sequentially up to end_orig.
    Keep only frames whose original indices are in needed_orig_indices.

    Returns: dict orig_idx -> RGB uint8 torch tensor (H,W,3)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_orig))
    current = int(start_orig)

    kept: Dict[int, torch.Tensor] = {}
    try:
        while current <= end_orig:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            if current in needed_orig_indices:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                kept[current] = torch.from_numpy(frame_rgb)  # uint8 (H,W,3)

            current += 1
    finally:
        cap.release()

    return kept


def build_clip_from_needed_frames(
    needed_frames: Dict[int, torch.Tensor],
    keep_every_n: int,
    clip_start_15fps: int,
    clip_len: int = 16,
) -> torch.Tensor:
    """
    Build one clip (3,16,224,224) float32 in [0,1] from sparse decoded frames.
    """
    wanted_orig = [(clip_start_15fps + j) * keep_every_n for j in range(clip_len)]
    frames_out: List[torch.Tensor] = []
    for orig_idx in wanted_orig:
        if orig_idx not in needed_frames:
            raise RuntimeError(f"Missing decoded frame orig_idx={orig_idx} (video keep_every_n={keep_every_n})")
        frames_out.append(_resize_center_crop_224(needed_frames[orig_idx]))

    clip_tchw = torch.stack(frames_out, dim=0)              # (16,3,224,224)
    clip = clip_tchw.permute(1, 0, 2, 3).contiguous()       # (3,16,224,224)
    return clip



# Stage-1 modes

@dataclass
class Stage1MetaBase:
    stage: str = "stage1_features"
    version: str = "dman_v21"
    backbone: str = "MCG-NJU/videomae-base"
    fps_target: int = 15
    clip_len: int = 16
    stride: int = 8
    global_pool: str = "mean(tokens)"
    patch_pool: str = "mean_time(reshape(8,196,768))"
    dtype_disk: str = "float16"


def _ucf_raw_video_path(repo_root: str, rel_path: str) -> str:
    # Your dataset layout: data/ucf_crime/raw/<Group>/<Video>.mp4
    return os.path.join(repo_root, "data/ucf_crime/raw", rel_path)


def dump_global_debug(
    repo_root: str,
    videos_jsonl: str,
    split_txt: str,
    out_dir: str,
    split_name: str,
    limit_videos: int,
    max_clips_per_video: int,
    batch_size: int,
    seed: int,
    device: str,
) -> str:
    rows = read_jsonl(videos_jsonl)
    split_set = set(read_split_txt(split_txt))

    chosen = [r for r in rows if r.get("split") == split_name and r.get("rel_path") in split_set]
    chosen = chosen[:max(0, limit_videos)]

    if len(chosen) == 0:
        raise RuntimeError(f"No videos found for split={split_name} in split list.")

    model, processor, _ = load_videomae(device=device)

    clip_keys: List[str] = []
    video_ids: List[str] = []
    clip_starts: List[int] = []
    labels: List[int] = []
    global_chunks: List[torch.Tensor] = []

    clip_batch: List[torch.Tensor] = []
    meta_batch: List[Tuple[str, int, int]] = []  # (video_id, clip_start, label_bin)

    print(f"[global] videos={len(chosen)} max_clips_per_video={max_clips_per_video} batch_size={batch_size}", flush=True)

    for v_i, rec in enumerate(chosen, start=1):
        video_id = rec["rel_path"]
        label_bin = int(rec["label_bin"])
        keep_every_n = int(rec["keep_every_n"])

        starts = deterministic_sample(rec["clip_starts"], video_id, max_clips_per_video, seed)
        if len(starts) == 0:
            continue

        video_path = _ucf_raw_video_path(repo_root, video_id)
        if not os.path.exists(video_path):
            print(f"[global] missing file: {video_id} (skip)", flush=True)
            continue

        # decode only what we need for this video (sparse sequential)
        needed_orig: Set[int] = set()
        for s in starts:
            for j in range(16):
                needed_orig.add((int(s) + j) * keep_every_n)

        start_orig = min(needed_orig)
        end_orig = max(needed_orig)

        print(f"[global] {v_i}/{len(chosen)} video={video_id} clips={len(starts)} orig_range=[{start_orig},{end_orig}]", flush=True)

        needed_frames = decode_needed_frames_sequential(
            video_path=video_path,
            start_orig=start_orig,
            end_orig=end_orig,
            needed_orig_indices=needed_orig,
        )

        for s in starts:
            clip = build_clip_from_needed_frames(
                needed_frames=needed_frames,
                keep_every_n=keep_every_n,
                clip_start_15fps=int(s),
                clip_len=16,
            )
            clip_batch.append(clip)
            meta_batch.append((video_id, int(s), label_bin))

            if len(clip_batch) == batch_size:
                clips_bcthw = torch.stack(clip_batch, dim=0)
                global_f32, _ = extract_batch(model, processor, clips_bcthw)
                global_f16 = global_f32.detach().cpu().to(torch.float16)

                for (vid, cs, lb) in meta_batch:
                    clip_keys.append(make_clip_key(vid, cs))
                    video_ids.append(vid)
                    clip_starts.append(cs)
                    labels.append(lb)

                global_chunks.append(global_f16)
                clip_batch = []
                meta_batch = []

        del needed_frames

    if len(clip_batch) > 0:
        clips_bcthw = torch.stack(clip_batch, dim=0)
        global_f32, _ = extract_batch(model, processor, clips_bcthw)
        global_f16 = global_f32.detach().cpu().to(torch.float16)

        for (vid, cs, lb) in meta_batch:
            clip_keys.append(make_clip_key(vid, cs))
            video_ids.append(vid)
            clip_starts.append(cs)
            labels.append(lb)

        global_chunks.append(global_f16)

    global_mat = torch.cat(global_chunks, dim=0) if global_chunks else torch.empty((0, 768), dtype=torch.float16)

    base = Stage1MetaBase()
    meta = {
        **base.__dict__,
        "split": split_name,
        "created_utc": utc_iso(),
        "git_commit": git_commit_or_unknown(repo_root),
        "config_hash": f"seed={seed}|limit_videos={limit_videos}|max_clips_per_video={max_clips_per_video}|batch_size={batch_size}",
    }

    payload = {
        "meta": meta,
        "clip_keys": clip_keys,
        "video_ids": video_ids,
        "clip_starts": torch.tensor(clip_starts, dtype=torch.int32),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "global": global_mat,
    }

    out_path = os.path.join(out_dir, f"global_{split_name}_debug_10vids.pt")
    atomic_save_pt(payload, out_path)
    return out_path


def dump_patch_normal_mem_shards(
    repo_root: str,
    videos_jsonl: str,
    train_split_txt: str,
    out_dir: str,
    limit_videos: int,
    max_clips_per_video: int,
    batch_size: int,
    shard_size: int,
    seed: int,
    device: str,
) -> List[str]:
    """
    Patch features for Normal-train only, sharded and memory-bounded.

    Output shard schema (locked):
      {
        "meta": {..., "split":"train", "subset":"normal_mem_budget"},
        "clip_keys": [...],
        "video_ids": [...],
        "clip_starts": int32[N],
        "patch": float16[N,196,768]
      }
    """
    rows = read_jsonl(videos_jsonl)
    train_set = set(read_split_txt(train_split_txt))

    normal_rows = [
        r for r in rows
        if r.get("split") == "train"
        and int(r.get("label_bin", 1)) == 0
        and r.get("rel_path") in train_set
    ]
    if limit_videos > 0:
        normal_rows = normal_rows[:limit_videos]

    if len(normal_rows) == 0:
        raise RuntimeError("No Normal-train videos found (label_bin==0).")

    os.makedirs(out_dir, exist_ok=True)
    model, processor, _ = load_videomae(device=device)

    base = Stage1MetaBase()

    shard_paths: List[str] = []
    shard_id = 0

    buf_clip_keys: List[str] = []
    buf_video_ids: List[str] = []
    buf_clip_starts: List[int] = []
    buf_patch_chunks: List[torch.Tensor] = []

    clip_batch: List[torch.Tensor] = []
    meta_batch: List[Tuple[str, int]] = []  # (video_id, clip_start)

    def flush_shard() -> None:
        nonlocal shard_id, buf_clip_keys, buf_video_ids, buf_clip_starts, buf_patch_chunks

        if len(buf_clip_keys) == 0:
            return

        patch_mat = torch.cat(buf_patch_chunks, dim=0) if buf_patch_chunks else torch.empty((0, 196, 768), dtype=torch.float16)
        patch_mat = patch_mat.to(torch.float16)

        meta = {
            **base.__dict__,
            "split": "train",
            "subset": "normal_mem_budget",
            "created_utc": utc_iso(),
            "git_commit": git_commit_or_unknown(repo_root),
            "config_hash": f"seed={seed}|limit_videos={limit_videos}|max_clips_per_video={max_clips_per_video}|batch_size={batch_size}|shard_size={shard_size}",
        }

        payload = {
            "meta": meta,
            "clip_keys": buf_clip_keys,
            "video_ids": buf_video_ids,
            "clip_starts": torch.tensor(buf_clip_starts, dtype=torch.int32),
            "patch": patch_mat,
        }

        out_path = os.path.join(out_dir, f"patch_normal_train_mem_shard_{shard_id:03d}.pt")
        atomic_save_pt(payload, out_path)
        shard_paths.append(out_path)

        shard_id += 1
        buf_clip_keys = []
        buf_video_ids = []
        buf_clip_starts = []
        buf_patch_chunks = []

    print(
        f"[patch_mem] videos={len(normal_rows)} max_clips_per_video={max_clips_per_video} "
        f"batch_size={batch_size} shard_size={shard_size}",
        flush=True
    )

    for v_i, rec in enumerate(normal_rows, start=1):
        video_id = rec["rel_path"]
        keep_every_n = int(rec["keep_every_n"])

        starts = deterministic_sample(rec["clip_starts"], video_id, max_clips_per_video, seed)
        if len(starts) == 0:
            continue

        video_path = _ucf_raw_video_path(repo_root, video_id)
        if not os.path.exists(video_path):
            print(f"[patch_mem] missing file: {video_id} (skip)", flush=True)
            continue

        # compute sparse frame set needed for this video’s chosen clips
        needed_orig: Set[int] = set()
        for s in starts:
            for j in range(16):
                needed_orig.add((int(s) + j) * keep_every_n)

        start_orig = min(needed_orig)
        end_orig = max(needed_orig)

        print(f"[patch_mem] {v_i}/{len(normal_rows)} video={video_id} clips={len(starts)} orig_range=[{start_orig},{end_orig}]", flush=True)

        needed_frames = decode_needed_frames_sequential(
            video_path=video_path,
            start_orig=start_orig,
            end_orig=end_orig,
            needed_orig_indices=needed_orig,
        )

        # build clips, run backbone in mini-batches, and flush shards
        for s in starts:
            clip = build_clip_from_needed_frames(
                needed_frames=needed_frames,
                keep_every_n=keep_every_n,
                clip_start_15fps=int(s),
                clip_len=16,
            )
            clip_batch.append(clip)
            meta_batch.append((video_id, int(s)))

            if len(clip_batch) == batch_size:
                clips_bcthw = torch.stack(clip_batch, dim=0)
                _, patch_f32 = extract_batch(model, processor, clips_bcthw)
                patch_f16 = patch_f32.detach().cpu().to(torch.float16)

                for (vid, cs) in meta_batch:
                    buf_clip_keys.append(make_clip_key(vid, cs))
                    buf_video_ids.append(vid)
                    buf_clip_starts.append(cs)

                buf_patch_chunks.append(patch_f16)

                clip_batch = []
                meta_batch = []

                # shard flush to bound RAM
                if len(buf_clip_keys) >= shard_size:
                    print(f"[patch_mem] flush shard {shard_id:03d} clips={len(buf_clip_keys)}", flush=True)
                    flush_shard()

        del needed_frames

    # flush remaining mini-batch
    if len(clip_batch) > 0:
        clips_bcthw = torch.stack(clip_batch, dim=0)
        _, patch_f32 = extract_batch(model, processor, clips_bcthw)
        patch_f16 = patch_f32.detach().cpu().to(torch.float16)

        for (vid, cs) in meta_batch:
            buf_clip_keys.append(make_clip_key(vid, cs))
            buf_video_ids.append(vid)
            buf_clip_starts.append(cs)

        buf_patch_chunks.append(patch_f16)

    # final shard flush
    if len(buf_clip_keys) > 0:
        print(f"[patch_mem] final flush shard {shard_id:03d} clips={len(buf_clip_keys)}", flush=True)
        flush_shard()

    return shard_paths


def dump_global_sharded(
    repo_root: str,
    videos_jsonl: str,
    split_txt: str,
    out_dir: str,
    split_name: str,
    limit_videos: int,
    max_clips_per_video: int,
    batch_size: int,
    shard_size: int,
    seed: int,
    device: str,
) -> List[str]:
    rows = read_jsonl(videos_jsonl)
    split_set = set(read_split_txt(split_txt))

    split_rows = [
        r for r in rows
        if r.get("split") == split_name
        and r.get("rel_path") in split_set
    ]
    if limit_videos > 0:
        split_rows = split_rows[:limit_videos]

    if len(split_rows) == 0:
        raise RuntimeError(f"No videos found for split={split_name} using split file.")

    os.makedirs(out_dir, exist_ok=True)
    model, processor, _ = load_videomae(device=device)

    base = Stage1MetaBase()

    shard_paths: List[str] = []
    shard_id = 0

    buf_clip_keys: List[str] = []
    buf_video_ids: List[str] = []
    buf_clip_starts: List[int] = []
    buf_labels: List[int] = []
    buf_global_chunks: List[torch.Tensor] = []

    clip_batch: List[torch.Tensor] = []
    meta_batch: List[Tuple[str, int, int]] = []  # (video_id, clip_start, label_bin)

    def flush_shard() -> None:
        nonlocal shard_id, buf_clip_keys, buf_video_ids, buf_clip_starts, buf_labels, buf_global_chunks

        if len(buf_clip_keys) == 0:
            return

        global_mat = torch.cat(buf_global_chunks, dim=0).to(torch.float16)  # (N,768)

        meta = {
            **base.__dict__,
            "split": split_name,
            "created_utc": utc_iso(),
            "git_commit": git_commit_or_unknown(repo_root),
            "config_hash": f"seed={seed}|limit_videos={limit_videos}|max_clips_per_video={max_clips_per_video}|batch_size={batch_size}|shard_size={shard_size}",
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
        buf_global_chunks = []

    print(
        f"[global_sharded] split={split_name} videos={len(split_rows)} "
        f"max_clips_per_video={max_clips_per_video} batch_size={batch_size} shard_size={shard_size}",
        flush=True,
    )

    for v_i, rec in enumerate(split_rows, start=1):
        video_id = rec["rel_path"]
        label_bin = int(rec["label_bin"])
        keep_every_n = int(rec["keep_every_n"])

        starts = deterministic_sample(rec["clip_starts"], video_id, max_clips_per_video, seed)
        if len(starts) == 0:
            continue

        video_path = _ucf_raw_video_path(repo_root, video_id)
        if not os.path.exists(video_path):
            print(f"[global_sharded] missing file: {video_id} (skip)", flush=True)
            continue

        needed_orig: Set[int] = set()
        for s in starts:
            for j in range(16):
                needed_orig.add((int(s) + j) * keep_every_n)

        start_orig = min(needed_orig)
        end_orig = max(needed_orig)

        print(
            f"[global_sharded] {v_i}/{len(split_rows)} video={video_id} clips={len(starts)} orig_range=[{start_orig},{end_orig}]",
            flush=True,
        )

        needed_frames = decode_needed_frames_sequential(
            video_path=video_path,
            start_orig=start_orig,
            end_orig=end_orig,
            needed_orig_indices=needed_orig,
        )

        for s in starts:
            clip = build_clip_from_needed_frames(
                needed_frames=needed_frames,
                keep_every_n=keep_every_n,
                clip_start_15fps=int(s),
                clip_len=16,
            )
            clip_batch.append(clip)
            meta_batch.append((video_id, int(s), label_bin))

            if len(clip_batch) == batch_size:
                clips_bcthw = torch.stack(clip_batch, dim=0)
                global_f32, _ = extract_batch(model, processor, clips_bcthw)
                global_f16 = global_f32.detach().cpu().to(torch.float16)

                for (vid, cs, lb) in meta_batch:
                    buf_clip_keys.append(make_clip_key(vid, cs))
                    buf_video_ids.append(vid)
                    buf_clip_starts.append(cs)
                    buf_labels.append(lb)

                buf_global_chunks.append(global_f16)

                clip_batch = []
                meta_batch = []

                if len(buf_clip_keys) >= shard_size:
                    print(f"[global_sharded] flush shard {shard_id:03d} clips={len(buf_clip_keys)}", flush=True)
                    flush_shard()

        del needed_frames

    # flush remainder batch
    if len(clip_batch) > 0:
        clips_bcthw = torch.stack(clip_batch, dim=0)
        global_f32, _ = extract_batch(model, processor, clips_bcthw)
        global_f16 = global_f32.detach().cpu().to(torch.float16)

        for (vid, cs, lb) in meta_batch:
            buf_clip_keys.append(make_clip_key(vid, cs))
            buf_video_ids.append(vid)
            buf_clip_starts.append(cs)
            buf_labels.append(lb)

        buf_global_chunks.append(global_f16)

    if len(buf_clip_keys) > 0:
        print(f"[global_sharded] final flush shard {shard_id:03d} clips={len(buf_clip_keys)}", flush=True)
        flush_shard()

    return shard_paths



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["global", "patch_normal_mem", "global_sharded"])

    # split is only used for global mode; patch mode always uses train + label_bin==0
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])

    # debug controls
    parser.add_argument("--limit_videos", type=int, default=10)

    # sampling + batching
    parser.add_argument("--max_clips_per_video", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)

    # patch mode memory control
    parser.add_argument("--shard_size", type=int, default=512)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    videos_jsonl = os.path.join(repo_root, "data/ucf_crime/artifacts/dman_v21/clip_index/videos.jsonl")
    split_txt = os.path.join(repo_root, f"data/ucf_crime/splits/dman_v21/{args.split}.txt")
    out_dir = os.path.join(repo_root, "data/ucf_crime/artifacts/dman_v21/features")

    if args.mode == "global":
        out_path = dump_global_debug(
            repo_root=repo_root,
            videos_jsonl=videos_jsonl,
            split_txt=split_txt,
            out_dir=out_dir,
            split_name=args.split,
            limit_videos=args.limit_videos,
            max_clips_per_video=args.max_clips_per_video,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        print(f"\n Wrote global debug shard: {out_path}")
        return

    elif args.mode == "global_sharded":
        shard_paths = dump_global_sharded(
            repo_root=repo_root,
            videos_jsonl=videos_jsonl,
            split_txt=split_txt,
            out_dir=out_dir,
            split_name=args.split,
            limit_videos=args.limit_videos,
            max_clips_per_video=args.max_clips_per_video,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            seed=args.seed,
            device=args.device,
        )

        print("\n Wrote global shards:")
        for p in shard_paths[:3]:
            print(" -", p)
        if len(shard_paths) > 3:
            print(f" - ... ({len(shard_paths)} shards total)")
        return


    # patch_normal_mem
    # always use train split list for patch mode (normal train videos only)
    elif args.mode == "patch_normal_mem":
        train_split_txt = os.path.join(repo_root, "data/ucf_crime/splits/dman_v21/train.txt")

        shard_paths = dump_patch_normal_mem_shards(
            repo_root=repo_root,
            videos_jsonl=videos_jsonl,
            train_split_txt=train_split_txt,
            out_dir=out_dir,
            limit_videos=args.limit_videos,
            max_clips_per_video=args.max_clips_per_video,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            seed=args.seed,
            device=args.device,
        )

        print("\n Wrote patch memory shards:")
        for p in shard_paths:
            print(" -", p)


if __name__ == "__main__":
    main()
