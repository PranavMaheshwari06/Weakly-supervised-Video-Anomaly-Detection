from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from stage2_class.src.io.fixed_bag_manifest import (
    build_fixed_bag_manifest,
    load_fixed_bag_sampler,
    write_fixed_bag_manifest,
)
from stage2_class.src.data.mg_mil_sampler import stable_sampling_seed
from stage2_class.src.data.mil_index import ClipRef
from stage2_class.src.eval.m29_eval_dispatch import compute_metrics
from stage2_class.src.utils.provenance import require_empty_run_directory


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


class StableSamplingTests(unittest.TestCase):
    def test_stable_seed_changes_with_context(self) -> None:
        first = stable_sampling_seed(1337, 4, "train", "video-a")
        self.assertEqual(first, stable_sampling_seed(1337, 4, "train", "video-a"))
        self.assertNotEqual(first, stable_sampling_seed(1337, 4, "fixed_eval:val", "video-a"))

    def test_selection_matches_across_python_processes(self) -> None:
        script = """
import json
from stage2_class.src.data.mg_mil_sampler import MgMilConfig, MgMilSampler
from stage2_class.src.data.mil_index import ClipRef
refs = [ClipRef(f'clip-{i}', i * 8, float(i), 'unused.pt', i) for i in range(40)]
sampler = MgMilSampler(MgMilConfig(), seed=1337)
selected = sampler.select_refs_for_video(refs, epoch=5, video_id='video-a')
print(json.dumps([ref.clip_key for ref in selected]))
"""
        root = Path(__file__).resolve().parents[2]
        outputs = []
        for hash_seed in ("1", "999"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = hash_seed
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(json.loads(result.stdout))
        self.assertEqual(outputs[0], outputs[1])


class FixedBagManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video_ids = ["video-a", "video-b"]
        self.video_to_clips = {
            "video-a": make_refs("video-a", 24),
            "video-b": make_refs("video-b", 9),
        }
        self.source_identity = {
            "root": "feature-store",
            "split": "val",
            "files": [{"path": "global_val_shard_00000.pt", "sha256": "abc"}],
            "identity_sha256": "feature-identity",
        }

    def write_manifest(self, directory: str) -> tuple[Path, dict]:
        payload = build_fixed_bag_manifest(
            split="val",
            video_ids=self.video_ids,
            video_to_clips=self.video_to_clips,
            bag_size=16,
            manifest_seed=20260831,
            source_feature_store=self.source_identity,
        )
        path = Path(directory) / "val.json"
        write_fixed_bag_manifest(path, payload)
        return path, payload

    def test_load_preserves_exact_order_and_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, payload = self.write_manifest(directory)
            sampler = load_fixed_bag_sampler(
                path=path,
                split="val",
                bag_size=16,
                manifest_seed=20260831,
                video_ids=self.video_ids,
                video_to_clips=self.video_to_clips,
                expected_source_identity="feature-identity",
            )
            for video_id in self.video_ids:
                selected = sampler.select_refs_for_video([], epoch=99, video_id=video_id)
                self.assertEqual([ref.clip_key for ref in selected], payload["bags"][video_id])
                self.assertEqual(len(selected), 16)

    def test_missing_manifest_clip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, payload = self.write_manifest(directory)
            missing_key = payload["bags"]["video-a"][0]
            changed = {
                video_id: [ref for ref in refs if ref.clip_key != missing_key]
                for video_id, refs in self.video_to_clips.items()
            }
            with self.assertRaisesRegex(ValueError, "missing from feature store"):
                load_fixed_bag_sampler(
                    path, "val", 16, 20260831, self.video_ids, changed, "feature-identity"
                )

    def test_clip_from_different_video_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, payload = self.write_manifest(directory)
            selected_key = payload["bags"]["video-a"][0]
            changed = {
                video_id: [ref for ref in refs if ref.clip_key != selected_key]
                for video_id, refs in self.video_to_clips.items()
            }
            selected_ref = next(
                ref for ref in self.video_to_clips["video-a"] if ref.clip_key == selected_key
            )
            changed["video-b"].append(selected_ref)
            with self.assertRaisesRegex(ValueError, "clip/video mismatch"):
                load_fixed_bag_sampler(
                    path, "val", 16, 20260831, self.video_ids, changed, "feature-identity"
                )


class OutputSafetyTests(unittest.TestCase):
    def test_non_empty_run_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "existing-run"
            run_dir.mkdir()
            (run_dir / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "unique run tag"):
                require_empty_run_directory(run_dir)


class EvaluatorMetricTests(unittest.TestCase):
    def test_synthetic_dispatch_and_binary_metrics(self) -> None:
        targets_dispatch = [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ]
        preds_dispatch = [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]
        metrics = compute_metrics(
            targets_dispatch,
            preds_dispatch,
            targets_binary=[0, 0, 1, 1],
            scores_binary=[0.1, 0.8, 0.9, 0.2],
            preds_binary=[0, 1, 1, 0],
        )

        self.assertAlmostEqual(metrics["macro_f1"], 13 / 18)
        self.assertAlmostEqual(metrics["macro_recall"], 2 / 3)
        self.assertEqual(metrics["binary_precision"], 0.5)
        self.assertEqual(metrics["binary_recall"], 0.5)
        self.assertEqual(metrics["binary_f1"], 0.5)
        self.assertEqual(metrics["binary_accuracy"], 0.5)
        self.assertEqual(metrics["binary_auroc"], 0.75)
        self.assertEqual(
            metrics["binary_confusion"],
            {"tp": 1, "fp": 1, "fn": 1, "tn": 1},
        )
        self.assertEqual(
            metrics["dispatch_prediction_distribution"],
            {"0,0,1": 1, "0,1,0": 1, "1,0,0": 1, "1,1,0": 1},
        )
        self.assertEqual(metrics["binary_prediction_distribution"], {"0": 2, "1": 2})

        precise_binary_score = 0.12345678901234568
        precise_dispatch_score = 0.9876543210987654
        payload = {
            "metrics": metrics,
            "binary_score": precise_binary_score,
            "dispatch_scores": [precise_dispatch_score],
        }
        serialized = json.dumps(payload, allow_nan=False)
        restored = json.loads(serialized)
        self.assertEqual(restored["binary_score"], precise_binary_score)
        self.assertEqual(restored["dispatch_scores"][0], precise_dispatch_score)

    def test_official_json_serialization_rejects_non_finite_values(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    json.dumps({"score": value}, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
