"""Create or validate fixed validation and test bag manifests without reading labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stage2_class.src.data.mil_index import build_clip_lookup_for_split
from stage2_class.src.io.fixed_bag_manifest import (
    build_fixed_bag_manifest,
    load_fixed_bag_sampler,
    manifest_path,
    refs_from_clip_lookup,
    write_fixed_bag_manifest,
)
from stage2_class.src.io.split_reader import read_split_txt
from stage2_class.src.utils.provenance import feature_store_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="stage2_class/configs/hierA_dispatch_v1.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    store_root = cfg["paths"]["feature_store_root"]
    splits_root = Path(cfg["paths"]["splits_root"])
    expected_dim = int(cfg["stage1"]["global_dim"])
    bag_size = int(cfg["stage2"]["bag_size"])
    evaluation_seed = int(cfg["reproducibility"]["evaluation_seed"])
    output_pattern = cfg["reproducibility"]["evaluation_bag_manifest_pattern"]

    for split in ("val", "test"):
        video_ids = read_split_txt(str(splits_root / f"{split}.txt"))
        clip_lookup = build_clip_lookup_for_split(store_root, split, expected_dim, -1)
        video_to_clips = refs_from_clip_lookup(clip_lookup)
        source_identity = feature_store_identity(store_root, split)
        output_path = manifest_path(output_pattern, split, bag_size, evaluation_seed)

        if not output_path.exists():
            payload = build_fixed_bag_manifest(
                split=split,
                video_ids=video_ids,
                video_to_clips=video_to_clips,
                bag_size=bag_size,
                manifest_seed=evaluation_seed,
                source_feature_store=source_identity,
            )
            write_fixed_bag_manifest(output_path, payload)

        load_fixed_bag_sampler(
            path=output_path,
            split=split,
            bag_size=bag_size,
            manifest_seed=evaluation_seed,
            video_ids=video_ids,
            video_to_clips=video_to_clips,
            expected_source_identity=source_identity["identity_sha256"],
        )
        print(f"Fixed {split} bags ready: {output_path}")


if __name__ == "__main__":
    main()
