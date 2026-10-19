"""Independent mask/rank/count verification of Phase 5 evidence.

Does not import the production metric/evaluation code. Release is allowed only
after every expected record, identity, metric and aggregate passes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import rankdata
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
import torch

PHASE4 = Path('reproducibility/temporal_protocol')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def compare(actual, expected, where='root'):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), (where, set(actual) ^ set(expected))
        for key in expected:
            compare(actual[key], expected[key], where + '.' + key)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), where
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{where}[{i}]')
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), (where, actual, expected)
    else:
        assert actual == expected, (where, actual, expected)


def mask(intervals, total):
    result = np.zeros(total, dtype=bool)
    for a, b in intervals:
        assert 0 <= a < b <= total
        result[a:b] = True
    return result


def intervals(bits):
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    return list(map(list, zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())))


def support(starts, meta):
    n, total, length = meta['keep_every_n'], meta['src_frame_count'], meta['clip_len']
    assert n == max(1, round(meta['src_fps'] / 15)) and length == 16 and meta['stride'] == 8
    assert starts == sorted(set(starts))
    assert all((s + 15) * n < total for s in starts)
    return [(s * n, min((s + 16) * n, total)) for s in starts]


def predicted(scores, starts, meta, threshold):
    values = np.asarray(scores, dtype=np.float64)
    pieces = np.split(values, np.flatnonzero(np.diff(starts) != 8) + 1)
    smoothed = np.concatenate([gaussian_filter1d(p, sigma=2, mode='reflect') for p in pieces])
    spread = float(np.ptp(smoothed))
    scaled = (smoothed - smoothed.min()) / spread if spread else np.zeros_like(smoothed)
    result = np.zeros(meta['src_frame_count'], dtype=bool)
    for (a, b), keep in zip(support(starts, meta), scaled >= threshold):
        if keep:
            result[a:b] = True
    return result


def average_precision(gt, scores):
    order = np.argsort(-scores, kind='stable')
    truth, values = gt[order], scores[order]
    boundaries = np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values) - 1]
    positives = np.cumsum(truth)[boundaries]
    increments = np.diff(np.r_[0, positives])
    return float(np.sum(increments * positives / (boundaries + 1)) / int(gt.sum()))


def temporal(record, scores, whole=False):
    meta, starts, keys = record['metadata'], record['clip_starts'], record['clip_keys']
    total = meta['src_frame_count']
    truth = record['intervals']['ground_truth']
    selected = np.ones(total, dtype=bool) if whole else predicted(scores, starts, meta, .5)
    top = None if whole else sorted(range(len(keys)), key=lambda i: (-scores[i], starts[i], keys[i]))[0]
    result = {'ap': None, 'top_hit': None, 'iou': None, 'zero_iou': None,
              'null_reason': None if record['annotation_status'] == 'anomaly' else record['annotation_status'],
              'top_clip_key': keys[top] if top is not None else None, 'top_overlap_frames': None,
              'predicted_intervals': intervals(selected)}
    if not truth:
        return result
    gt = mask(truth, total)
    projected = np.ones(total) if whole else np.full(total, min(scores) - 1, dtype=np.float64)
    supports = support(starts, meta)
    if not whole:
        for (a, b), score in zip(supports, scores):
            projected[a:b] = np.maximum(projected[a:b], score)
    result['ap'] = average_precision(gt, projected)
    result['iou'] = int((selected & gt).sum()) / int((selected | gt).sum())
    result['zero_iou'] = not bool((selected & gt).any())
    if top is not None:
        a, b = supports[top]
        result['top_overlap_frames'] = int(gt[a:b].sum())
        result['top_hit'] = bool(gt[a:b].any())
    return result


def temporal_aggregate(values):
    if not values:
        return {'videos': 0, 'null_reason': 'no eligible observations'}
    result = {'videos': len(values)}
    for key in ('ap', 'top_hit', 'iou', 'zero_iou'):
        selected = [float(v[key]) for v in values if v[key] is not None]
        result[key + '_mean'] = statistics.mean(selected) if selected else None
    overlaps = [v['iou'] for v in values]
    result.update(iou_median=statistics.median(overlaps), zero_iou_count=overlaps.count(0),
                  iou_at_least={str(t): sum(v >= t for v in overlaps) / len(overlaps) for t in (.1, .3, .5, .7)})
    return result


def routing_aggregate(rows, field):
    y = np.asarray([r['dispatch_target'] for r in rows])
    if field == 'all_services':
        p = np.ones_like(y)
    elif field == 'binary_gated_all_services':
        p = np.repeat(np.asarray([r['binary_pred'] for r in rows])[:, None], 3, axis=1)
    else:
        p = np.asarray([r[field] for r in rows])
    precision, recall, f1, counts = precision_recall_fscore_support(y, p, average=None, zero_division=0)
    confusion = multilabel_confusion_matrix(y, p)
    channels = {}
    for i, name in enumerate(('police', 'medical', 'fire')):
        tn, fp, fn, tp = confusion[i].ravel().tolist()
        channels[name] = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'support': int(counts[i]),
                          'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
    counts_by_vector = Counter(''.join(str(v) for v in r) for r in p.tolist())
    return {'videos': len(rows), 'channels': channels, 'macro_f1': statistics.mean(f1.tolist()),
            'hamming_loss': np.count_nonzero(y != p) / y.size,
            'exact_match': sum(np.array_equal(a, b) for a, b in zip(y, p)) / len(rows),
            'predicted_vector_counts': {format(i, '03b'): counts_by_vector[format(i, '03b')] for i in range(8)}}


def aggregate(records):
    eligible = [r for r in records if r['annotation_status'] == 'anomaly']
    normal = [r for r in records if r['annotation_status'] == 'normal']
    rows = [r['routing'] for r in records]
    conditional = [r for r in rows if r['binary_true']]
    assert (len(records), len(eligible), len(normal), len(conditional)) == (290, 135, 150, 140)
    result = {'temporal': temporal_aggregate([r['temporal_metrics']['primary'] for r in eligible]), 'category_temporal': {},
              'routing': {}, 'cad_priority_counts': {k: sum(r['cad_priority'] == k for r in rows)
                    for k in ('CLEAR', 'P1-EMERGENCY', 'P1-MEDICAL', 'P2-POLICE', 'P3-REVIEW')}}
    for category in sorted({r['metadata']['group'] for r in records}):
        result['category_temporal'][category] = temporal_aggregate([r['temporal_metrics']['primary'] for r in eligible if r['metadata']['group'] == category])
    truth = np.asarray([r['binary_true'] for r in rows])
    predictions = np.asarray([r['binary_pred'] for r in rows])
    ranks = rankdata([r['binary_score'] for r in rows])
    positives, negatives = int(truth.sum()), int((1 - truth).sum())
    auc = (ranks[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    pr, re, f1, _ = precision_recall_fscore_support(truth, predictions, average='binary', zero_division=0)
    result['binary'] = {'videos': len(rows), 'auroc': float(auc), 'accuracy': float(np.mean(truth == predictions)),
                        'tp': int(np.sum((truth == 1) & (predictions == 1))), 'fp': int(np.sum((truth == 0) & (predictions == 1))),
                        'fn': int(np.sum((truth == 1) & (predictions == 0))), 'tn': int(np.sum((truth == 0) & (predictions == 0))),
                        'precision': float(pr), 'recall': float(re), 'f1': float(f1)}
    display = [r['temporal_metrics']['display_gated_iou'] for r in eligible]
    result['display_gated'] = {'videos': len(display), 'iou_mean': statistics.mean(display), 'iou_median': statistics.median(display),
                               'zero_iou_count': display.count(0), 'zero_iou_frequency': display.count(0) / len(display)}
    result['category_display_gated'] = {}
    for category in sorted({r['metadata']['group'] for r in records}):
        subset = [r['temporal_metrics']['display_gated_iou'] for r in eligible if r['metadata']['group'] == category]
        result['category_display_gated'][category] = ({'videos': len(subset), 'iou_mean': statistics.mean(subset),
            'iou_median': statistics.median(subset), 'zero_iou_count': subset.count(0), 'zero_iou_frequency': subset.count(0) / len(subset)}
            if subset else {'videos': 0, 'null_reason': 'no eligible observations'})
    result['normal_temporal'] = {}
    for scope in ('primary_ungated', 'display_gated'):
        fractions = [float(mask(r['intervals'][scope], r['metadata']['src_frame_count']).mean()) for r in normal]
        result['normal_temporal'][scope] = {'videos': len(normal), 'support_fraction_mean': statistics.mean(fractions),
                                           'any_interval_frequency': sum(v > 0 for v in fractions) / len(fractions)}
    for scope in ('primary', 'demo'):
        result['routing'][scope] = {'raw': routing_aggregate(rows, 'raw_' + scope), 'gated': routing_aggregate(rows, 'gated_' + scope),
                                    'conditional': routing_aggregate(conditional, 'raw_' + scope)}
    result['routing_baselines'] = {name: {b: routing_aggregate(subset, b) for b in ('all_services', 'binary_gated_all_services')}
                                   for name, subset in [('all', rows), ('conditional', conditional)]}
    return result


def sample_statistics(values):
    if all(isinstance(v, dict) for v in values):
        keys = set().union(*(v.keys() for v in values))
        return {k: sample_statistics([v.get(k) for v in values]) for k in sorted(keys)}
    if all(isinstance(v, (int, float)) for v in values):
        return {'values': values, 'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values), 'n': 3}
    return {'values': values, 'mean': None, 'sample_sd': None}


def verify_record(record, audit, candidate, teacher, validator):
    if validator is not None:
        validator.validate(record)
    assert not record['failures'] and record['status'] != 'technical_failure'
    assert record['video_id'] == audit['video_id'] == candidate['video_id']
    assert record['clip_keys'] == candidate['clip_keys'] and record['clip_starts'] == candidate['clip_starts']
    assert record['annotation_status'] == audit['annotation_status']
    assert record['intervals']['ground_truth'] == audit['gt_intervals_half_open']
    assert record['status'] == ('temporal_excluded' if audit['gt_intervals_half_open'] is None else 'ok')
    compare(record['metadata'], {**audit['metadata'], 'group': audit['group']})
    keys, starts, meta = record['clip_keys'], record['clip_starts'], record['metadata']
    assert len(keys) == len(record['attention']) == len(record['teacher_scores'])
    assert len(keys) == len(set(keys)) and np.isfinite(record['attention']).all()
    assert math.isclose(sum(record['attention']), 1, abs_tol=2e-6)
    assert record['teacher_scores'] == [teacher[k] for k in keys]
    compare(record['temporal_metrics']['primary'], temporal(record, record['attention']))
    for name in ('whole_video', 'uniform_clips', 'random_clips', 'teacher_distance'):
        if name == 'random_clips':
            seed = int.from_bytes(hashlib.sha256(('temporal_system_v1:' + record['video_id']).encode('utf-8')).digest()[:8], byteorder='big')
            saliency = np.random.Generator(np.random.PCG64(seed)).random(len(keys), dtype=np.float64)
            saliency /= saliency.sum()
        elif name == 'teacher_distance':
            saliency = record['teacher_scores']
        else:
            saliency = np.full(len(keys), 1 if name == 'whole_video' else 1 / len(keys))
        compare(record['baseline_metrics'][name], temporal(record, saliency, name == 'whole_video'))
    total = meta['src_frame_count']
    primary = predicted(record['attention'], starts, meta, .5)
    display = predicted(record['attention'], starts, meta, .65)
    routing = record['routing']
    logits = np.asarray(routing['binary_logits'])
    dispatch_logits = np.asarray(routing['dispatch_logits'])
    assert logits.shape == (2,) and dispatch_logits.shape == (3,) and np.isfinite(logits).all() and np.isfinite(dispatch_logits).all()
    binary_score = np.exp(logits - logits.max()); binary_score /= binary_score.sum()
    assert abs(binary_score[1] - routing['binary_score']) <= 2e-7
    np.testing.assert_allclose(routing['dispatch_scores'], 1 / (1 + np.exp(-dispatch_logits)), atol=2e-7, rtol=0)
    binary = int(np.argmax(logits))
    assert routing['binary_pred'] == binary and routing['binary_true'] == audit['label_bin']
    # Explicit frozen project mapping, independently transcribed from the contract.
    mapping = {'Arson': [1, 0, 1], 'Explosion': [0, 1, 1], 'RoadAccidents': [1, 1, 1],
               **{k: [1, 1, 0] for k in ('Assault', 'Abuse', 'Fighting', 'Shooting')},
               **{k: [1, 0, 0] for k in ('Burglary', 'Robbery', 'Shoplifting', 'Stealing', 'Vandalism', 'Arrest')}}
    assert routing['dispatch_target'] == (mapping[audit['group']] if audit['label_bin'] else [0, 0, 0])
    for name, thresholds in [('primary', [.5] * 3), ('demo', [.65, .60, .65])]:
        raw = [int(x >= t) for x, t in zip(routing['dispatch_scores'], thresholds)]
        assert routing['raw_' + name] == raw
        assert routing['gated_' + name] == (raw if binary else [0, 0, 0])
    gated = display if binary else np.zeros(total, dtype=bool)
    for name, bits in [('primary_ungated', primary), ('display_ungated', display), ('display_gated', gated)]:
        assert record['intervals'][name] == intervals(bits)
    for field, bits in [('normal_primary_support_fraction', primary), ('normal_display_gated_support_fraction', gated)]:
        compare(record['temporal_metrics'][field], float(bits.mean()) if record['annotation_status'] == 'normal' else None)
    services = [name for name, enabled in zip(('police', 'medical', 'fire'), routing['gated_demo']) if enabled]
    priority = 'CLEAR' if not binary else 'P1-EMERGENCY' if 'fire' in services else 'P1-MEDICAL' if 'medical' in services else 'P2-POLICE' if 'police' in services else 'P3-REVIEW'
    assert routing['cad_services'] == services and routing['cad_priority'] == priority and routing['alert_triggered'] == bool(binary)
    trigger = int(np.flatnonzero(gated)[0]) if gated.any() else None
    assert routing['trigger_frame'] == trigger
    assert routing['trigger_seconds'] == (trigger / meta['src_fps'] if trigger is not None else None)
    alert = routing['cad_alert']
    assert alert['priority'] == priority and alert['dispatch'] == services and alert['alert_triggered'] == bool(binary)
    if gated.any():
        a, b = intervals(gated)[0]
        compare(alert['anomaly_window'], {'start_frame': a, 'end_frame': b - 1, 'start_sec': round(a / meta['src_fps'], 2), 'end_sec': round(b / meta['src_fps'], 2)})
        assert alert['alert_latency_sec'] == round(trigger / meta['src_fps'], 2)
    else:
        assert alert['anomaly_window'] is None and 'alert_latency_sec' not in alert
    truth = record['intervals']['ground_truth']
    if truth:
        gt = mask(truth, total)
        overlap = int((gt & gated).sum()) / int((gt | gated).sum())
        compare(record['temporal_metrics']['display_gated_iou'], overlap)
        assert record['temporal_metrics']['display_gated_zero_iou'] == (overlap == 0)
    else:
        assert record['temporal_metrics']['display_gated_iou'] is None and record['temporal_metrics']['display_gated_zero_iou'] is None
    cov = mask(support(starts, meta), total)
    gaps = intervals(~cov)
    measured = {'clips': len(starts), 'unique_clips': len(set(starts)), 'covered_source_frames': int(cov.sum()),
                'fraction': float(cov.mean()), 'max_uncovered_gap_frames': max((b - a for a, b in gaps), default=0),
                'annotated_frames_covered': int((cov & mask(truth, total)).sum()) if truth else None,
                'annotation_coverage': int((cov & mask(truth, total)).sum()) / int(mask(truth, total).sum()) if truth else None}
    expected_coverage = {**measured, 'expected_clips': len(keys), 'realized_clips': len(keys),
                         'scopes': {'dense_realized': measured, 'sparse': audit['coverage']['sparse'], 'fixed_bag': audit['coverage']['fixed_bag']}}
    compare(record['coverage'], expected_coverage)


def coverage_aggregate(records):
    result = {}
    for scope in ('fixed_bag', 'sparse', 'dense_realized'):
        values = [r['coverage']['scopes'][scope] for r in records]
        fractions = [v['fraction'] for v in values]
        annotated = [v['annotation_coverage'] for v in values if v['annotation_coverage'] is not None]
        total = sum(r['metadata']['src_frame_count'] for r in records)
        covered = sum(v['covered_source_frames'] for v in values)
        result[scope] = {'videos': len(records), 'clips': sum(v['clips'] for v in values), 'video_mean_coverage': statistics.mean(fractions),
                    'video_median_coverage': statistics.median(fractions), 'min_coverage': min(fractions),
                    'total_source_frames': total, 'covered_source_frames': covered, 'frame_weighted_coverage': covered / total,
                    'annotation_eligible_videos': len(annotated),
                    'anomaly_mean_annotation_coverage': statistics.mean(annotated) if annotated else None,
                    'annotation_null_reason': None if annotated else 'no eligible annotations',
                    'anomalies_with_zero_annotation_coverage': annotated.count(0)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', action='store_true', help='Exclusively create canonical summary after all checks pass')
    parser.add_argument('--attempt-id', choices=['attempt_001', 'attempt_002'], default='attempt_001')
    # Size and mtime of the raw videos only match on the machine that ran the extraction.
    parser.add_argument('--skip-raw-video-check', action='store_true')
    args = parser.parse_args()
    if args.skip_raw_video_check:
        print('SKIP: raw-video size/mtime check against the extraction records')
    protocol = load(PHASE4 / 'protocol.json')
    assert sha(PHASE4 / 'protocol.json') == '6a3757b21650f995a216e4b88cfc916c2ff875a39de478019268528a18d9e5a0'
    for path, expected in {**protocol['input_hashes'], **protocol['source_hashes'], **protocol['evidence_hashes']}.items():
        # The frozen protocol records its companion files under their original folder.
        path = path.replace('final_thesis_execution/phase_04_temporal_design_compute_preflight', str(PHASE4))
        assert sha(path) == expected, path
    paths = protocol['outputs']
    attempt = Path(paths['run_root']) / args.attempt_id
    extraction = load(attempt / 'extraction_complete.json')
    candidate_path = Path(extraction.get('candidate_manifest_path', attempt / 'candidate_manifest.json'))
    summary_path = attempt / 'evaluation/provisional_summary.json'
    summary = load(summary_path)
    assert summary['failures'] == [] and summary['population'] == protocol['evaluation_population']
    assert not list(attempt.rglob('failure.json'))
    candidates = load(candidate_path)
    audit = load(PHASE4 / 'coverage_audit.json')['videos']
    ids = Path('data/ucf_crime/splits/dman_v21/test.txt').read_text().splitlines()
    assert [r['video_id'] for r in candidates] == [r['video_id'] for r in audit] == ids
    teacher_path = Path(paths['teacher_root']) / 'teacher_scores_test.pt'
    assert sha(teacher_path) == summary['teacher_sha256']
    teacher_obj = torch.load(teacher_path, map_location='cpu', weights_only=False)
    teacher = dict(zip(teacher_obj['clip_keys'], teacher_obj['scores'].tolist()))
    assert list(teacher) == [k for r in candidates for k in r['clip_keys']] and len(teacher) == 69090
    torch.set_num_threads(4)
    prototypes = torch.load('artifacts/stage2/memory_lora/global_prototypes_k100.pt', map_location='cpu', weights_only=False)['prototypes'].double().numpy()
    files, realized = [], []
    max_teacher_error = 0.
    for path in sorted(Path(paths['feature_store']).glob('global_test_shard_*.pt')):
        files.append({'path': path.name, 'sha256': sha(path)})
        shard = torch.load(path, map_location='cpu', weights_only=False)
        assert shard['global'].dtype == torch.float16 and len(shard['clip_keys']) <= 512
        x = shard['global'].double().numpy()
        assert x.shape == (len(shard['clip_keys']), 768) and np.isfinite(x).all()
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        distances = []
        for offset in range(0, len(x), 32):
            distances.extend(np.square(x[offset:offset + 32, None] - prototypes[None]).sum(axis=2).min(axis=1).tolist())
        error = np.max(np.abs(np.asarray(distances) - np.asarray([teacher[k] for k in shard['clip_keys']])))
        max_teacher_error = max(max_teacher_error, float(error))
        assert error <= 2e-6, 'Teacher differs from independent float64 squared distances'
        for vid, start, key in zip(shard['video_ids'], shard['clip_starts'].tolist(), shard['clip_keys']):
            assert key == f'{vid}__s{start:06d}'
        realized.extend(shard['clip_keys'])
    assert realized == list(teacher)
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    compare(summary['feature_store_identity'], {'root': paths['feature_store'], 'split': 'test', 'files': files, 'identity_sha256': identity})
    store_manifest = load(Path(paths['feature_store']) / 'manifest.json')
    assert store_manifest['candidate_manifest_sha256'] == sha(candidate_path)
    compare(store_manifest['identity'], summary['feature_store_identity'])
    extraction = load(attempt / 'extraction_complete.json')
    assert extraction['clips'] == 69090 and extraction['videos'] == 290 and not extraction['failures']
    assert extraction['feature_store_identity'] == summary['feature_store_identity']
    assert extraction['teacher_sha256'] == summary['teacher_sha256']
    assert store_manifest['validation_replay_sha256'] == sha(attempt / 'validation_replay.json')
    assert store_manifest['execution_sha256'] == sha(attempt / 'execution.json')
    execution = load(attempt / 'execution.json')
    assert execution['source_sha256'] == sha('utils/extract_temporal_system.py')
    assert execution['protocol_sha256'] == summary['protocol_sha256']
    if args.attempt_id != protocol['attempt_id']:
        recovery = load(attempt / 'recovery_plan.json')
        assert recovery['protocol_sha256'] == summary['protocol_sha256']
        assert recovery['candidate_manifest_sha256'] == sha(candidate_path)
        assert recovery['retained_clips'] == extraction['retained_clips'] == 11264
        assert extraction['new_clips'] == 57826
        assert store_manifest['recovery_plan_sha256'] == sha(attempt / 'recovery_plan.json')
        assert execution['recovery_source_sha256'] == sha('utils/resume_temporal_extraction.py')
        for path, expected in recovery['previous_attempt_files'].items():
            assert sha(path) == expected, ('Previous attempt changed', path)
        for retained in recovery['retained_shards']:
            assert sha(Path(paths['feature_store']) / retained['path']) == retained['sha256']
    evaluation_execution = load(attempt / 'evaluation/execution.json')
    for path, expected in evaluation_execution['source_hashes'].items():
        assert sha(path) == expected, ('Evaluator changed since execution', path)
    replay = load(attempt / 'validation_replay.json')
    assert replay['passed'] and replay['loaded_backbone_state_sha256'] == load(PHASE4 / 'loader_teacher_audit.json')['loaded_backbone_state_sha256']
    assert len(replay['features']) == 18 and all(r['max_abs_error'] <= .005 and r['cosine_similarity'] >= .99999 for r in replay['features'])
    assert replay['teacher_validation_clips'] == 17117 and replay['teacher_max_abs_difference'] <= .001
    for i, candidate in enumerate(candidates):
        video_paths = extraction.get('video_record_paths')
        evidence = load(video_paths[i] if video_paths else attempt / f'extraction_video_{i:03d}.json')
        assert evidence['realized_clip_keys'] == candidate['clip_keys']
        for key in ('video_id', 'clip_keys', 'clip_starts', 'metadata'):
            assert evidence[key] == candidate[key]
        if not args.skip_raw_video_check:
            source = (Path('data/ucf_crime/raw') / candidate['video_id']).stat()
            assert evidence['raw_bytes'] == source.st_size and evidence['raw_mtime_ns'] == source.st_mtime_ns
    try:
        from jsonschema import Draft202012Validator
        validator = Draft202012Validator(load(PHASE4 / 'PER_VIDEO_SCHEMA.json'))
    except ImportError:
        validator = None
    recomputed, first_records = {}, None
    assert len(summary['record_files']) == 870
    seen_paths = set()
    for checkpoint in protocol['checkpoints']:
        tag = checkpoint['run_tag']
        entries = [r for r in summary['record_files'] if Path(r['path']).parent.name == tag]
        assert len(entries) == 290
        records = []
        for number, (entry, row, candidate) in enumerate(zip(entries, audit, candidates)):
            assert entry['path'] not in seen_paths
            seen_paths.add(entry['path'])
            assert Path(entry['path']) == attempt / 'evaluation' / tag / f'video_{number:03d}.json'
            assert sha(entry['path']) == entry['sha256']
            record = load(entry['path'])
            assert record['checkpoint'] == tag
            assert record['protocol_id'] == protocol['protocol_id']
            assert set(load(PHASE4 / 'PER_VIDEO_SCHEMA.json')['required']) <= set(record)
            compare(record['provenance'], {'checkpoint_sha256': checkpoint['sha256'], 'feature_store_identity': identity,
                    'teacher_sha256': summary['teacher_sha256'], 'protocol_sha256': sha(PHASE4 / 'protocol.json'), 'attempt_id': args.attempt_id})
            verify_record(record, row, candidate, teacher, validator)
            if first_records is not None:
                assert record['baseline_metrics'] == first_records[number]['baseline_metrics']
                assert record['coverage'] == first_records[number]['coverage']
            records.append(record)
        recomputed[tag] = aggregate(records)
        compare(summary['checkpoint_summaries'][tag], recomputed[tag], tag)
        if first_records is None:
            first_records = records
        print(f'PASS independent metrics, routing, CAD and aggregates: {tag}', flush=True)
    compare(summary['three_checkpoint_statistics'], sample_statistics(list(recomputed.values())))
    baseline_summaries = {}
    eligible = [r for r in first_records if r['annotation_status'] == 'anomaly']
    for name in protocol['metrics']['temporal_baselines']:
        baseline_summaries[name] = {'overall': temporal_aggregate([r['baseline_metrics'][name] for r in eligible]),
            'by_category': {g: temporal_aggregate([r['baseline_metrics'][name] for r in eligible if r['metadata']['group'] == g])
                            for g in sorted({r['metadata']['group'] for r in first_records})}, 'replicates': 1, 'checkpoint_independent': True}
    compare(summary['temporal_baselines'], baseline_summaries)
    compare(summary['coverage'], coverage_aggregate(first_records))
    compare(summary['category_coverage'], {g: coverage_aggregate([r for r in first_records if r['metadata']['group'] == g])
            for g in sorted({r['metadata']['group'] for r in first_records})})
    validation = {'status': 'pass', 'records': 870, 'checkpoints': 3, 'clips': 69090, 'technical_failures': 0,
                  'verifier_sha256': sha(__file__), 'provisional_summary_sha256': sha(summary_path),
                  'schema_validation': 'jsonschema Draft202012' if validator else 'required fields plus independent semantic checks',
                  'max_teacher_error_independent_float64': max_teacher_error}
    if args.release:
        destination = Path(paths['summary'])
        if destination.exists():
            raise FileExistsError(destination)
        validation_path = attempt / 'independent_verification.json'
        with validation_path.open('x') as f:
            json.dump(validation, f, indent=2, allow_nan=False); f.write('\n')
        destination.parent.mkdir(parents=True, exist_ok=False)
        summary['status'] = 'complete_independently_verified'
        summary['independent_verification'] = {'path': str(validation_path), 'sha256': sha(validation_path)}
        with destination.open('x') as f:
            json.dump(summary, f, indent=2, allow_nan=False); f.write('\n')
    else:
        canonical = Path(paths['summary'])
        if canonical.exists():
            released = load(canonical)
            verification = released.pop('independent_verification')
            assert sha(verification['path']) == verification['sha256']
            assert released.pop('status') == 'complete_independently_verified'
            expected = dict(summary); expected.pop('status')
            compare(released, expected)
    print(json.dumps(validation), flush=True)


if __name__ == '__main__':
    main()
