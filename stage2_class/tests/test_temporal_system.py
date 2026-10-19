"""Hand-computable edge cases for the frozen Phase 5 metrics and output guards."""
import tempfile
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from stage2_class.src.eval.temporal_system_metrics import (
    coverage, iou, random_scores, routing_metrics, temporal_metrics,
    checkpoint_summary, three_checkpoint_statistics,
)
from stage2_class.src.eval.temporal_geometry import attention_intervals
from utils.evaluate_temporal_system import routing_record, coverage_summary
from utils.verify_temporal_system import aggregate, compare, verify_record, sample_statistics, coverage_aggregate
from stage3_dispatch.src.alert_router import generate_alert
from utils.extract_temporal_system import teacher_scores, write_json, save_tensor


class TemporalSystemTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(src_fps=15, keep_every_n=1, src_frame_count=48, clip_len=16, stride=8)

    def test_tied_ap_and_top_tie(self):
        # All 32 covered frames share one threshold: AP is 8/32, not 8/48.
        result = temporal_metrics([1, 1], [0, 16], ['a', 'b'], self.meta, [(0, 8)])
        self.assertEqual(result['ap'], .25)
        self.assertTrue(result['top_hit'])
        self.assertEqual(result['top_clip_key'], 'a')
        self.assertEqual(result['top_overlap_frames'], 8)
        self.assertEqual(result['predicted_intervals'], [])
        self.assertEqual(result['iou'], 0)

    def test_uncovered_positive_and_overlap_projection(self):
        # Highest-scoring 16 frames contain all 8 positives, so AP = 1/2.
        result = temporal_metrics([.2, .8], [0, 8], ['a', 'b'], self.meta, [(8, 16)])
        self.assertEqual(result['ap'], .5)
        # Positives only outside support are ranked after every covered frame.
        result = temporal_metrics([1, 1], [0, 16], ['a', 'b'], self.meta, [(40, 48)])
        self.assertAlmostEqual(result['ap'], 8 / 48)
        self.assertFalse(result['top_hit'])

    def test_whole_video_and_normal_null(self):
        result = temporal_metrics([1], [0], ['a'], self.meta, [(0, 12)], whole_video=True)
        self.assertEqual(result['ap'], .25)
        self.assertEqual(result['iou'], .25)
        self.assertIsNone(result['top_hit'])
        normal = temporal_metrics([1], [0], ['a'], self.meta, [], 'normal')
        self.assertIsNone(normal['ap'])
        self.assertEqual(normal['null_reason'], 'normal')
        self.assertIsNone(iou([], []))
        self.assertEqual(iou([], [(1, 2)]), 0)

    def test_coverage_union_and_gap(self):
        result = coverage([0, 8], self.meta, [(16, 32)])
        self.assertEqual(result['covered_source_frames'], 24)
        self.assertEqual(result['max_uncovered_gap_frames'], 24)
        self.assertEqual(result['annotation_coverage'], .5)
        self.assertEqual(coverage([], self.meta, [])['max_uncovered_gap_frames'], 48)

    def test_routing_empty_predictions_and_counts(self):
        result = routing_metrics([[1, 0, 1], [0, 0, 0]], [[0, 0, 0], [1, 0, 0]])
        self.assertEqual(result['channels']['medical']['f1'], 0)
        self.assertEqual(result['channels']['police']['fn'], 1)
        self.assertEqual(result['channels']['police']['fp'], 1)
        self.assertEqual(result['hamming_loss'], .5)
        self.assertEqual(result['predicted_vector_counts']['000'], 1)
        self.assertEqual(result['predicted_vector_counts']['100'], 1)
        self.assertEqual(result['predicted_vector_counts']['111'], 0)

    def test_random_reuse_and_sample_sd(self):
        np.testing.assert_array_equal(random_scores('video', 10), random_scores('video', 10))
        self.assertFalse(np.array_equal(random_scores('video', 10), random_scores('other', 10)))
        self.assertAlmostEqual(random_scores('video', 10).sum(), 1)
        self.assertEqual(three_checkpoint_statistics([1., 2., 3.])['sample_sd'], 1)

    def test_failures_block_aggregation(self):
        with self.assertRaises(ValueError):
            checkpoint_summary([])
        with self.assertRaises(ValueError):
            temporal_metrics([np.nan], [0], ['a'], self.meta, [(0, 1)])

    def test_teacher_no_prototype_renormalization(self):
        x = torch.tensor([[3., 4.], [0., 0.]], dtype=torch.float16)
        p = torch.tensor([[1., 0.], [0., .5]], dtype=torch.float16)
        expected = ((torch.nn.functional.normalize(x.float(), dim=1)[:, None] - p.float()[None]) ** 2).sum(2).min(1).values
        torch.testing.assert_close(teacher_scores(x, p), expected)

    def test_output_collision(self):
        with tempfile.TemporaryDirectory() as d:
            for name, writer, value in [('x.json', write_json, {}), ('x.pt', save_tensor, torch.zeros(1))]:
                path = Path(d) / name
                writer(path, value)
                before = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    writer(path, value)
                self.assertEqual(path.read_bytes(), before)

    def test_cad_without_interval_keeps_gate_and_priority(self):
        for services, priority in [([], 'P3-REVIEW'), (['police'], 'P2-POLICE'),
                                    (['police', 'medical'], 'P1-MEDICAL'), (['fire', 'medical'], 'P1-EMERGENCY')]:
            alert = generate_alert({'binary_anomaly_detected': True, 'dispatch_services': services, 'anomaly_intervals': []})
            self.assertTrue(alert['alert_triggered'])
            self.assertEqual(alert['priority'], priority)
            self.assertIsNone(alert['anomaly_window'])
            self.assertNotIn('alert_latency_sec', alert)
        alert = generate_alert({'binary_anomaly_detected': False, 'dispatch_services': ['fire']})
        self.assertEqual(alert['priority'], 'CLEAR')
        self.assertEqual(alert['dispatch'], [])

    def test_independent_records_and_aggregates(self):
        # Full declared denominators, with synthetic predictions and annotations.
        # This checks cross-component behavior without reading test predictions.
        records = []
        for number in range(290):
            vid = f'synthetic_{number}'
            starts, keys = [0, 8, 16, 24], [f'{vid}__s{s:06d}' for s in (0, 8, 16, 24)]
            normal = number >= 140
            truth = [] if normal else [(4, 20)] if number < 135 else None
            status = 'normal' if normal else 'anomaly' if truth else 'excluded'
            row = {'video_id': vid, 'metadata': self.meta, 'group': 'Normal' if normal else 'Arson',
                   'label_bin': int(not normal), 'annotation_status': status, 'gt_intervals_half_open': truth}
            scores = [.1, .2, .3, .4] if number % 2 else [.25] * 4
            teacher = [.5, .1, .9, .2]
            reason = None if truth else status
            primary = temporal_metrics(scores, starts, keys, self.meta, truth, reason)
            display = attention_intervals(scores, starts, self.meta, .65)
            output = SimpleNamespace(binary_logits=torch.tensor([[0., 0. if number % 3 else 1.]]),
                                     anomaly_logits=torch.tensor([[0., 1., -1.]]))
            routing, gated = routing_record(output, row, display)
            baselines = {name: temporal_metrics(saliency, starts, keys, self.meta, truth, reason, name == 'whole_video')
                         for name, saliency in [('whole_video', [1.] * 4), ('uniform_clips', [.25] * 4),
                                                ('random_clips', random_scores(vid, 4)), ('teacher_distance', teacher)]}
            cov = coverage(starts, self.meta, truth)
            row['coverage'] = {'fixed_bag': cov, 'sparse': cov}
            record = {'video_id': vid, 'status': 'temporal_excluded' if truth is None else 'ok',
                      'annotation_status': status, 'metadata': {**self.meta, 'group': row['group']},
                      'clip_keys': keys, 'clip_starts': starts, 'attention': scores, 'teacher_scores': teacher,
                      'intervals': {'ground_truth': truth, 'primary_ungated': primary['predicted_intervals'],
                                    'display_ungated': display, 'display_gated': gated},
                      'temporal_metrics': {'primary': primary, 'display_gated_iou': iou(gated, truth) if truth else None,
                                           'display_gated_zero_iou': iou(gated, truth) == 0 if truth else None,
                                           'normal_primary_support_fraction': sum(b-a for a,b in primary['predicted_intervals']) / self.meta['src_frame_count'] if normal else None,
                                           'normal_display_gated_support_fraction': sum(b-a for a,b in gated) / self.meta['src_frame_count'] if normal else None},
                      'baseline_metrics': baselines, 'routing': routing, 'failures': [],
                      'coverage': {**cov, 'expected_clips': 4, 'realized_clips': 4,
                                   'scopes': {'fixed_bag': cov, 'sparse': cov, 'dense_realized': cov}}}
            # Match persisted JSON interval representation.
            record, row = json.loads(json.dumps(record)), json.loads(json.dumps(row))
            candidate = {'video_id': vid, 'clip_keys': keys, 'clip_starts': starts}
            verify_record(record, row, candidate, dict(zip(keys, teacher)), None)
            if number == 0:
                corrupted = copy.deepcopy(record)
                corrupted['temporal_metrics']['primary']['ap'] += .1
                with self.assertRaises(AssertionError):
                    verify_record(corrupted, row, candidate, dict(zip(keys, teacher)), None)
            records.append(record)
        production = checkpoint_summary(records)
        independent = aggregate(records)
        compare(production, independent)
        compare(coverage_summary(records), coverage_aggregate(records))
        compare(coverage_summary(records[140:]), coverage_aggregate(records[140:]))
        compare(three_checkpoint_statistics([production] * 3), sample_statistics([independent] * 3))
        broken = copy.deepcopy(records)
        broken[0]['failures'] = [{'code': 'decode_failure'}]
        with self.assertRaises(ValueError):
            checkpoint_summary(broken)


if __name__ == '__main__':
    unittest.main()
