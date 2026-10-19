from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from stage2_class.src.data.mg_mil_sampler import MgMilSampler
from stage2_class.src.data.mil_index import ClipRef
from stage2_class.src.utils.provenance import sha256_json


SCHEMA_VERSION = 1


def selected_clip_keys(
    sampler: MgMilSampler,
    video_clips: List[ClipRef],
    epoch: int,
    video_id: str,
) -> List[str]:
    refs = sampler.select_refs_for_video(
        video_clips,
        epoch=epoch,
        video_id=video_id,
        context="train",
    )
    bag_size = sampler.cfg.bag_size
    while len(refs) < bag_size:
        refs.append(refs[-1])
    return [ref.clip_key for ref in refs[:bag_size]]


def build_training_bag_plan(
    video_ids: List[str],
    video_to_clips: Dict[str, List[ClipRef]],
    sampler: MgMilSampler,
    maximum_epochs: int,
    source_feature_identity: str,
    teacher_scores_sha256: str,
) -> Dict:
    if maximum_epochs < 1:
        raise ValueError("maximum_epochs must be at least 1")

    missing = [video_id for video_id in video_ids if video_id not in video_to_clips]
    if missing:
        raise ValueError(f"Training videos missing clips: {missing[:3]}")

    epoch_identities = []
    for epoch in range(maximum_epochs):
        bags = [
            [
                video_id,
                selected_clip_keys(sampler, video_to_clips[video_id], epoch, video_id),
            ]
            for video_id in video_ids
        ]
        epoch_identities.append({
            "epoch": epoch,
            "bags_sha256": sha256_json(bags),
        })

    metadata = {
        "training_seed": int(sampler.seed),
        "sampling_context": "train",
        "bag_size": int(sampler.cfg.bag_size),
        "warmup_epochs": int(sampler.cfg.warmup_epochs),
        "mixture_uniform_ratio": float(sampler.cfg.mixture_uniform_ratio),
        "top_fraction": float(sampler.cfg.top_fraction),
        "maximum_epochs": int(maximum_epochs),
        "eligible_video_count": len(video_ids),
        "bags_per_epoch": len(video_ids),
        "planned_bag_count": len(video_ids) * maximum_epochs,
        "video_ids_sha256": sha256_json(video_ids),
        "source_feature_identity_sha256": source_feature_identity,
        "teacher_scores_sha256": teacher_scores_sha256,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "epoch_identities": epoch_identities,
    }
    payload["plan_identity_sha256"] = sha256_json(payload)
    return payload


def write_training_bag_plan(path: str | Path, payload: Dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
