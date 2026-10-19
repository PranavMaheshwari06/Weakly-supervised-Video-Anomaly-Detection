"""Evaluate all three original checkpoints under temporal_system_v1.

Writes per-video evidence and a provisional aggregate. Only the independent
verifier may release the canonical summary. Never loads an ablation checkpoint.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from stage2_class.src.eval.temporal_geometry import attention_intervals, interval_length
from stage2_class.src.eval.temporal_system_metrics import (
    checkpoint_summary, coverage, iou, random_scores, temporal_metrics, temporal_summary,
    three_checkpoint_statistics,
)
from stage2_class.src.models.mil_hierarchical import HierarchicalAttentionMilHead
from stage2_class.src.utils.dispatch_label_map import DISPATCH_MAP, DISPATCH_CHANNELS
from stage2_class.src.utils.provenance import feature_store_identity, sha256_file
from stage3_dispatch.src.alert_router import generate_alert
from utils.extract_temporal_system import PHASE4, PROTOCOL_SHA, read_json, write_json


def feature_videos(store):
    """Hold one variable-length feature bag and one on-disk shard at a time."""
    current, keys, starts, features = None, [], [], []
    seen = set()
    for path in sorted(store.glob('global_test_shard_*.pt')):
        shard = torch.load(path, map_location='cpu', weights_only=False)
        assert shard['global'].dtype == torch.float16 and shard['global'].shape == (len(shard['clip_keys']), 768)
        assert torch.isfinite(shard['global']).all()
        for i, (vid, key, start) in enumerate(zip(shard['video_ids'], shard['clip_keys'], shard['clip_starts'].tolist())):
            if current is not None and vid != current:
                yield current, keys, starts, torch.stack(features).float()
                seen.add(current)
                keys, starts, features = [], [], []
            assert vid not in seen, 'Noncontiguous video in feature store'
            current = vid
            keys.append(key)
            starts.append(start)
            features.append(shard['global'][i].clone())
    if current is not None:
        yield current, keys, starts, torch.stack(features).float()


def routing_record(output, row, display):
    binary_logits = output.binary_logits[0].cpu()
    dispatch_logits = output.anomaly_logits[0].cpu()
    scores = torch.sigmoid(dispatch_logits).tolist()
    pred = int(binary_logits.argmax())
    primary = [int(s >= .5) for s in scores]
    demo = [int(s >= t) for s, t in zip(scores, (.65, .60, .65))]
    gated_demo = demo if pred else [0, 0, 0]
    effective = display if pred else []
    services = [name for name, active in zip(DISPATCH_CHANNELS, gated_demo) if active]
    trigger = effective[0][0] if effective else None
    metadata = {'video': row['video_id'], 'fps': row['metadata']['src_fps'],
                'total_frames': row['metadata']['src_frame_count'], 'binary_anomaly_detected': bool(pred),
                'dispatch_services': services, 'dispatch_scores': dict(zip(DISPATCH_CHANNELS, scores)),
                'anomaly_intervals': [{'start_frame': a, 'end_frame': b - 1} for a, b in effective],
                'alert_trigger_frame': trigger}
    alert = generate_alert(metadata)
    alert.pop('timestamp')  # Presentation timestamp has no deterministic scientific meaning.
    label = row['label_bin']
    if label and row['group'] not in DISPATCH_MAP:
        raise ValueError('Unknown anomalous dispatch category')
    result = {'binary_true': label, 'dispatch_target': DISPATCH_MAP[row['group']] if label else [0, 0, 0],
              'binary_logits': binary_logits.tolist(), 'binary_score': float(torch.softmax(binary_logits, dim=0)[1]),
              'binary_pred': pred, 'dispatch_logits': dispatch_logits.tolist(), 'dispatch_scores': scores,
              'raw_primary': primary, 'gated_primary': primary if pred else [0, 0, 0],
              'raw_demo': demo, 'gated_demo': gated_demo, 'cad_priority': alert['priority'],
              'alert_triggered': alert['alert_triggered'], 'cad_services': alert['dispatch'],
              'trigger_frame': trigger, 'trigger_seconds': trigger / row['metadata']['src_fps'] if trigger is not None else None,
              'cad_alert': alert}
    return result, effective


def coverage_summary(records):
    output = {}
    for scope in ('fixed_bag', 'sparse', 'dense_realized'):
        values = [r['coverage']['scopes'][scope] for r in records]
        fractions = [v['fraction'] for v in values]
        annotated = [v['annotation_coverage'] for v in values if v['annotation_coverage'] is not None]
        output[scope] = {'videos': len(values), 'clips': sum(v['clips'] for v in values),
                         'total_source_frames': sum(r['metadata']['src_frame_count'] for r in records),
                         'covered_source_frames': sum(v['covered_source_frames'] for v in values),
                         'video_mean_coverage': float(np.mean(fractions)), 'video_median_coverage': float(np.median(fractions)),
                         'min_coverage': min(fractions),
                         'frame_weighted_coverage': sum(v['covered_source_frames'] for v in values) / sum(r['metadata']['src_frame_count'] for r in records),
                         'annotation_eligible_videos': len(annotated),
                         'anomaly_mean_annotation_coverage': float(np.mean(annotated)) if annotated else None,
                         'annotation_null_reason': None if annotated else 'no eligible annotations',
                         'anomalies_with_zero_annotation_coverage': sum(v == 0 for v in annotated)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--attempt-id', choices=['attempt_001', 'attempt_002'], default='attempt_001')
    args = parser.parse_args()
    assert sha256_file(PHASE4 / 'protocol.json') == PROTOCOL_SHA
    subprocess.run([sys.executable, '-m', 'utils.verify_temporal_preflight'], check=True)
    protocol = read_json(PHASE4 / 'protocol.json')
    paths = protocol['outputs']
    attempt = Path(paths['run_root']) / args.attempt_id
    if args.attempt_id != protocol['attempt_id']:
        recovery = read_json(attempt / 'recovery_plan.json')
        assert recovery['protocol_sha256'] == PROTOCOL_SHA and recovery['retained_clips'] == 11264
    complete = read_json(attempt / 'extraction_complete.json')
    assert complete['status'] == 'complete' and not complete['failures']
    assert read_json(attempt / 'validation_replay.json')['passed']
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; request host execution')
    store = Path(paths['feature_store'])
    identity = feature_store_identity(store, 'test')
    assert identity == complete['feature_store_identity']
    teacher_path = Path(paths['teacher_root']) / 'teacher_scores_test.pt'
    teacher_sha = sha256_file(teacher_path)
    assert teacher_sha == complete['teacher_sha256']
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    assert len(teacher['clip_keys']) == len(set(teacher['clip_keys'])) == 69090
    lookup = dict(zip(teacher['clip_keys'], teacher['scores'].float().tolist()))
    candidates = read_json(complete.get('candidate_manifest_path', attempt / 'candidate_manifest.json'))
    expected_keys = [k for r in candidates for k in r['clip_keys']]
    assert teacher['clip_keys'] == expected_keys
    audit = read_json(PHASE4 / 'coverage_audit.json')['videos']
    assert [r['video_id'] for r in audit] == [r['video_id'] for r in candidates]
    destination = attempt / 'evaluation'
    destination.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    context = {'video_id': None, 'checkpoint': None, 'stage': 'MIL', 'affected_keys': [],
               'expected_count': 290, 'observed_count': 0, 'retry_identity': None}
    started = time.perf_counter()
    try:
        write_json(destination / 'execution.json', {'command': sys.argv, 'device': args.device,
                   'torch': torch.__version__, 'mil_dtype': 'float32', 'autocast': False, 'top_k_ratio': .25,
                   'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                   'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                   'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
                   'source_hashes': {p: sha256_file(p) for p in [__file__, 'stage2_class/src/eval/temporal_system_metrics.py']},
                   'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   'protocol_sha256': PROTOCOL_SHA, 'feature_store_identity': identity['identity_sha256'], 'teacher_sha256': teacher_sha})
        baseline_cache, summaries, record_files, first_records = {}, {}, [], None
        for checkpoint in protocol['checkpoints']:
            tag = checkpoint['run_tag']
            context['checkpoint'] = tag
            cfg = yaml.safe_load(Path(checkpoint['resolved_config']).read_text())
            assert cfg['stage2']['inject_teacher_score'] and cfg['stage2']['attention_hidden_dim'] == 256
            assert cfg['stage2']['num_dispatch_channels'] == 3 and cfg['stage2']['top_fraction'] == .25
            ckpt = torch.load(checkpoint['checkpoint'], map_location='cpu', weights_only=False)
            model = HierarchicalAttentionMilHead(769, 3, 256, cfg['stage2']['dropout'], .25)
            model.load_state_dict(ckpt['model_state'], strict=True)
            model.to(args.device).float().eval()
            records = []
            (destination / tag).mkdir(exist_ok=False)
            for number, (vid, keys, starts, features) in enumerate(feature_videos(store)):
                context.update(video_id=vid, affected_keys=keys, observed_count=len(records))
                row, candidate = audit[number], candidates[number]
                assert vid == row['video_id'] == candidate['video_id']
                assert keys == candidate['clip_keys'] and starts == candidate['clip_starts']
                assert len(set(keys)) == len(keys) and starts == sorted(set(starts))
                scores = torch.tensor([lookup[k] for k in keys], dtype=torch.float32)
                inputs = torch.cat([features, scores[:, None]], dim=1)[None].to(args.device)
                assert torch.isfinite(inputs).all() and inputs.shape == (1, len(keys), 769)
                with torch.inference_mode():
                    output = model(inputs, torch.ones((1, len(keys)), dtype=torch.bool, device=args.device))
                assert all(torch.isfinite(t).all() for t in [output.attention, output.binary_logits, output.anomaly_logits])
                attention = output.attention[0].cpu().tolist()
                meta, truth = row['metadata'], row['gt_intervals_half_open']
                reason = None if row['annotation_status'] == 'anomaly' else row['annotation_status']
                primary = temporal_metrics(attention, starts, keys, meta, truth, reason)
                display = attention_intervals(attention, starts, meta, .65)
                routing, gated_display = routing_record(output, row, display)
                if vid not in baseline_cache:
                    baselines = {}
                    for name, saliency in [('whole_video', np.ones(len(keys))), ('uniform_clips', np.full(len(keys), 1 / len(keys))),
                                            ('random_clips', random_scores(vid, len(keys))), ('teacher_distance', scores.numpy())]:
                        baselines[name] = temporal_metrics(saliency, starts, keys, meta, truth, reason, whole_video=name == 'whole_video')
                    baseline_cache[vid] = baselines
                dense = coverage(starts, meta, truth)
                assert dense == row['coverage']['dense_candidate'], 'Realized coverage differs from candidate support'
                cov = {**dense, 'expected_clips': len(candidate['clip_keys']), 'realized_clips': len(keys),
                       'scopes': {'fixed_bag': row['coverage']['fixed_bag'], 'sparse': row['coverage']['sparse'], 'dense_realized': dense}}
                record = {'protocol_id': protocol['protocol_id'], 'video_id': vid, 'checkpoint': tag,
                          'status': 'temporal_excluded' if truth is None else 'ok', 'annotation_status': row['annotation_status'],
                          'metadata': {**meta, 'group': row['group']}, 'clip_keys': keys, 'clip_starts': starts,
                          'attention': attention, 'teacher_scores': scores.tolist(),
                          'intervals': {'ground_truth': truth, 'primary_ungated': primary['predicted_intervals'],
                                        'display_ungated': display, 'display_gated': gated_display},
                          'temporal_metrics': {'primary': primary, 'display_gated_iou': iou(gated_display, truth) if truth else None,
                                               'display_gated_zero_iou': iou(gated_display, truth) == 0 if truth else None,
                                               'normal_primary_support_fraction': interval_length(primary['predicted_intervals']) / meta['src_frame_count'] if reason == 'normal' else None,
                                               'normal_display_gated_support_fraction': interval_length(gated_display) / meta['src_frame_count'] if reason == 'normal' else None},
                          'baseline_metrics': baseline_cache[vid], 'coverage': cov, 'routing': routing, 'failures': [],
                          'provenance': {'checkpoint_sha256': checkpoint['sha256'], 'feature_store_identity': identity['identity_sha256'],
                                         'teacher_sha256': teacher_sha, 'protocol_sha256': PROTOCOL_SHA, 'attempt_id': args.attempt_id}}
                path = destination / tag / f'video_{number:03d}.json'
                write_json(path, record)
                record_files.append({'path': str(path), 'sha256': sha256_file(path)})
                records.append(record)
            summaries[tag] = checkpoint_summary(records)
            if first_records is None:
                first_records = records
            print(f'Completed {tag}: {len(records)} video records', flush=True)
        baseline_summaries = {}
        eligible = [r for r in first_records if r['annotation_status'] == 'anomaly']
        for name in protocol['metrics']['temporal_baselines']:
            baseline_summaries[name] = {'overall': temporal_summary([r['baseline_metrics'][name] for r in eligible]),
                'by_category': {g: temporal_summary([r['baseline_metrics'][name] for r in eligible if r['metadata']['group'] == g])
                                for g in sorted({r['metadata']['group'] for r in first_records})},
                'replicates': 1, 'checkpoint_independent': True}
        summary = {'protocol_id': protocol['protocol_id'], 'status': 'awaiting_independent_verification',
                   'protocol_sha256': PROTOCOL_SHA, 'attempt_id': args.attempt_id,
                   'population': protocol['evaluation_population'], 'checkpoint_summaries': summaries,
                   'three_checkpoint_statistics': three_checkpoint_statistics(list(summaries.values())),
                   'temporal_baselines': baseline_summaries, 'coverage': coverage_summary(first_records),
                   'category_coverage': {g: coverage_summary([r for r in first_records if r['metadata']['group'] == g])
                                         for g in sorted({r['metadata']['group'] for r in first_records})},
                   'record_files': record_files, 'feature_store_identity': identity, 'teacher_sha256': teacher_sha,
                   'evaluation_seconds': time.perf_counter() - started, 'failures': []}
        write_json(destination / 'provisional_summary.json', summary)
    except BaseException as exc:
        write_json(destination / 'failure.json', {**context, 'code': 'evaluation_failure', 'message': repr(exc)})
        # Expected populations remain explicit even if an attempt stops early.
        unavailable = []
        for checkpoint in protocol['checkpoints']:
            for number, candidate in enumerate(candidates):
                record_path = destination / checkpoint['run_tag'] / f'video_{number:03d}.json'
                if not record_path.exists():
                    unavailable.append({'video_id': candidate['video_id'], 'checkpoint': checkpoint['run_tag'],
                                        'status': 'technical_failure', 'expected_keys': candidate['clip_keys'],
                                        'metrics': None, 'predictions': None,
                                        'reason': 'evaluation attempt stopped; see failure.json'})
        write_json(destination / 'unavailable_records.json', unavailable)
        raise


if __name__ == '__main__':
    main()
