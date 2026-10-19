from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from stage2_class.src.data.mg_mil_sampler import MgMilConfig, MgMilSampler
from stage2_class.src.data.mil_index import ClipRef
from stage2_class.src.utils.provenance import sha256_json


SCHEMA_VERSION = 1


class FixedBagSampler:
    def __init__(self, bag_size: int, refs_by_video: Dict[str, List[ClipRef]]) -> None:
        self.cfg = MgMilConfig(bag_size=bag_size)
        self.refs_by_video = refs_by_video

    def select_refs_for_video(
        self,
        video_clips: List[ClipRef],
        epoch: int,
        video_id: str,
        context: str = "fixed_eval",
    ) -> List[ClipRef]:
        del video_clips, epoch, context
        if video_id not in self.refs_by_video:
            raise KeyError(f"Fixed bag missing video_id={video_id}")
        return list(self.refs_by_video[video_id])


def manifest_path(path_pattern: str, split: str, bag_size: int, seed: int) -> Path:
    return Path(path_pattern.format(split=split, bag_size=bag_size, seed=seed))


def refs_from_clip_lookup(
    clip_lookup: Dict[str, Tuple[str, int, str, int]],
) -> Dict[str, List[ClipRef]]:
    video_to_clips: Dict[str, List[ClipRef]] = {}
    for clip_key, (video_id, clip_start, shard_path, local_index) in clip_lookup.items():
        ref = ClipRef(
            clip_key=clip_key,
            clip_start=int(clip_start),
            teacher_score=0.0,
            shard_path=shard_path,
            local_index=int(local_index),
        )
        video_to_clips.setdefault(video_id, []).append(ref)

    for refs in video_to_clips.values():
        refs.sort(key=lambda ref: (ref.clip_start, ref.clip_key))
    return video_to_clips


def build_fixed_bag_manifest(
    split: str,
    video_ids: List[str],
    video_to_clips: Dict[str, List[ClipRef]],
    bag_size: int,
    manifest_seed: int,
    source_feature_store: Dict,
) -> Dict:
    sampler = MgMilSampler(MgMilConfig(bag_size=bag_size, warmup_epochs=1), manifest_seed)
    bags: Dict[str, List[str]] = {}

    for video_id in video_ids:
        if video_id not in video_to_clips:
            raise ValueError(f"Cannot build fixed bag: video_id missing clips: {video_id}")
        refs = sampler.select_refs_for_video(
            video_to_clips[video_id],
            epoch=0,
            video_id=video_id,
            context=f"fixed_eval:{split}",
        )
        while len(refs) < bag_size:
            refs.append(refs[-1])
        bags[video_id] = [ref.clip_key for ref in refs[:bag_size]]

    metadata = {
        "split": split,
        "bag_size": int(bag_size),
        "manifest_seed": int(manifest_seed),
        "sampling_context": f"fixed_eval:{split}",
        "video_ids_sha256": sha256_json(video_ids),
        "bags_sha256": sha256_json(bags),
        "source_feature_store": source_feature_store,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "bags": bags,
    }
    metadata["manifest_sha256"] = sha256_json(payload)
    return payload


def write_fixed_bag_manifest(path: str | Path, payload: Dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_fixed_bag_manifest(
    path: str | Path,
    split: str,
    bag_size: int,
    manifest_seed: int,
    video_ids: List[str],
    expected_source_identity: str | None = None,
) -> Dict:
    manifest_file = Path(path)
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Fixed {split} bag manifest not found: {manifest_file}. "
            "Generate evaluation bags before training or evaluation."
        )

    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported bag manifest schema: {manifest_file}")

    metadata = payload.get("metadata", {})
    if metadata.get("split") != split:
        raise ValueError(f"Bag manifest split mismatch: expected {split}, got {metadata.get('split')}")
    if int(metadata.get("bag_size", -1)) != bag_size:
        raise ValueError("Bag manifest bag_size mismatch")
    if int(metadata.get("manifest_seed", -1)) != manifest_seed:
        raise ValueError("Bag manifest seed mismatch")
    if metadata.get("sampling_context") != f"fixed_eval:{split}":
        raise ValueError("Bag manifest sampling context mismatch")
    if metadata.get("video_ids_sha256") != sha256_json(video_ids):
        raise ValueError("Bag manifest video list does not match the requested split")

    recorded_checksum = metadata.get("manifest_sha256")
    metadata_without_checksum = dict(metadata)
    metadata_without_checksum.pop("manifest_sha256", None)
    checksum_payload = {
        "schema_version": payload["schema_version"],
        "metadata": metadata_without_checksum,
        "bags": payload.get("bags"),
    }
    if recorded_checksum != sha256_json(checksum_payload):
        raise ValueError(f"Bag manifest checksum mismatch: {manifest_file}")

    source = metadata.get("source_feature_store", {})
    if expected_source_identity and source.get("identity_sha256") != expected_source_identity:
        raise ValueError("Bag manifest feature-store identity mismatch")

    bags = payload.get("bags")
    if not isinstance(bags, dict) or set(bags) != set(video_ids):
        raise ValueError("Bag manifest must contain exactly one bag for every split video")
    for video_id, clip_keys in bags.items():
        if not isinstance(clip_keys, list) or len(clip_keys) != bag_size:
            raise ValueError(f"Fixed bag for {video_id} must contain exactly {bag_size} clip keys")
        if not all(isinstance(clip_key, str) for clip_key in clip_keys):
            raise ValueError(f"Fixed bag for {video_id} contains a non-string clip key")
    if metadata.get("bags_sha256") != sha256_json(bags):
        raise ValueError("Bag manifest bag checksum mismatch")
    return payload


def load_fixed_bag_sampler(
    path: str | Path,
    split: str,
    bag_size: int,
    manifest_seed: int,
    video_ids: List[str],
    video_to_clips: Dict[str, List[ClipRef]],
    expected_source_identity: str | None = None,
) -> FixedBagSampler:
    payload = read_fixed_bag_manifest(
        path=path,
        split=split,
        bag_size=bag_size,
        manifest_seed=manifest_seed,
        video_ids=video_ids,
        expected_source_identity=expected_source_identity,
    )

    all_refs = {
        ref.clip_key: (video_id, ref)
        for video_id, refs in video_to_clips.items()
        for ref in refs
    }
    refs_by_video: Dict[str, List[ClipRef]] = {}
    for video_id in video_ids:
        selected: List[ClipRef] = []
        for clip_key in payload["bags"][video_id]:
            match = all_refs.get(clip_key)
            if match is None:
                raise ValueError(f"Bag manifest clip is missing from feature store: {clip_key}")
            actual_video_id, ref = match
            if actual_video_id != video_id:
                raise ValueError(
                    f"Bag manifest clip/video mismatch: {clip_key} belongs to {actual_video_id}, "
                    f"not {video_id}"
                )
            selected.append(ref)
        if len(selected) != bag_size:
            raise ValueError(f"Fixed bag for {video_id} has {len(selected)} clips, expected {bag_size}")
        refs_by_video[video_id] = selected

    return FixedBagSampler(bag_size=bag_size, refs_by_video=refs_by_video)
