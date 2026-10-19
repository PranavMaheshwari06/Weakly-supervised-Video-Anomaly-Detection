from __future__ import annotations

import unittest

from stage2_class.src.data.mg_mil_sampler import MgMilConfig, MgMilSampler
from stage2_class.src.data.mil_index import ClipRef
from stage2_class.src.io.training_bag_plan import build_training_bag_plan


def make_refs(video_id: str, count: int) -> list[ClipRef]:
    return [
        ClipRef(
            clip_key=f"{video_id}:clip-{index:03d}",
            clip_start=index * 8,
            teacher_score=float(index),
            shard_path="unused.pt",
            local_index=index,
        )
        for index in range(count)
    ]


class TrainingBagPlanTests(unittest.TestCase):
    def test_plan_is_paired_by_seed_and_policy(self) -> None:
        video_ids = ["video-a", "video-b"]
        video_to_clips = {
            "video-a": make_refs("video-a", 24),
            "video-b": make_refs("video-b", 9),
        }
        guided = MgMilSampler(
            MgMilConfig(
                bag_size=16,
                warmup_epochs=2,
                mixture_uniform_ratio=0.5,
                top_fraction=0.25,
            ),
            seed=1337,
        )
        uniform = MgMilSampler(
            MgMilConfig(
                bag_size=16,
                warmup_epochs=2,
                mixture_uniform_ratio=1.0,
                top_fraction=0.0,
            ),
            seed=1337,
        )
        common = {
            "video_ids": video_ids,
            "video_to_clips": video_to_clips,
            "maximum_epochs": 4,
            "source_feature_identity": "features",
            "teacher_scores_sha256": "teacher",
        }
        guided_a = build_training_bag_plan(sampler=guided, **common)
        guided_b = build_training_bag_plan(sampler=guided, **common)
        uniform_plan = build_training_bag_plan(sampler=uniform, **common)

        self.assertEqual(guided_a, guided_b)
        self.assertNotEqual(
            guided_a["plan_identity_sha256"],
            uniform_plan["plan_identity_sha256"],
        )
        self.assertEqual(guided_a["metadata"]["eligible_video_count"], 2)
        self.assertEqual(guided_a["metadata"]["bags_per_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
