"""Metrics for the predeclared dense indexed system; no fitted parameters."""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from stage2_class.src.eval.temporal_geometry import (
    attention_intervals, clip_support, intersection_length, interval_length, merge_intervals,
)


def iou(predicted, truth):
    intersection = intersection_length(predicted, truth)
    union = interval_length(predicted) + interval_length(truth) - intersection
    return intersection / union if union else None


def temporal_metrics(scores, starts, keys, meta, truth, null_reason=None, whole_video=False):
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or scores.shape != (len(starts),) or not np.isfinite(scores).all():
        raise ValueError('Invalid saliency scores')
    if len(set(keys)) != len(keys) or len(keys) != len(starts):
        raise ValueError('Duplicate or mismatched keys')
    predicted = [(0, meta['src_frame_count'])] if whole_video else attention_intervals(scores, starts, meta, .5)
    top = None if whole_video else min(range(len(scores)), key=lambda i: (-scores[i], starts[i], keys[i]))
    result = {'ap': None, 'top_hit': None, 'iou': None, 'zero_iou': None,
              'null_reason': null_reason, 'top_clip_key': keys[top] if top is not None else None,
              'top_overlap_frames': None, 'predicted_intervals': predicted}
    if not truth:
        if null_reason is None:
            raise ValueError('Normal/excluded metrics need an explicit null reason')
        return result
    total = meta['src_frame_count']
    gt = np.zeros(total, dtype=bool)
    for start, end in truth:
        gt[start:end] = True
    projected = np.ones(total) if whole_video else np.full(total, scores.min() - 1, dtype=np.float64)
    if not whole_video:
        for start, score in zip(starts, scores):
            a, b = clip_support(start, meta)
            projected[a:b] = np.maximum(projected[a:b], score)
    result['ap'] = float(average_precision_score(gt, projected))
    result['iou'] = iou(predicted, truth)
    result['zero_iou'] = result['iou'] == 0
    if top is not None:
        overlap = intersection_length([clip_support(starts[top], meta)], truth)
        result.update(top_hit=overlap > 0, top_overlap_frames=overlap)
    return result


def random_scores(video_id, count):
    seed = int.from_bytes(hashlib.sha256(('temporal_system_v1:' + video_id).encode()).digest()[:8], 'big')
    scores = np.random.Generator(np.random.PCG64(seed)).random(count, dtype=np.float64)
    return scores / scores.sum()


def coverage(starts, meta, truth):
    support = merge_intervals([clip_support(s, meta) for s in sorted(set(starts))])
    total = meta['src_frame_count']
    boundaries = [(0, 0)] + support + [(total, total)]
    covered = interval_length(support)
    return {'clips': len(starts), 'unique_clips': len(set(starts)), 'covered_source_frames': covered,
            'fraction': covered / total,
            'max_uncovered_gap_frames': max(b[0] - a[1] for a, b in zip(boundaries[:-1], boundaries[1:])),
            'annotated_frames_covered': intersection_length(support, truth) if truth else None,
            'annotation_coverage': intersection_length(support, truth) / interval_length(truth) if truth else None}


def routing_metrics(targets, predictions):
    y, p = np.asarray(targets, dtype=int), np.asarray(predictions, dtype=int)
    if y.ndim != 2 or y.shape != p.shape or y.shape[1] != 3 or not len(y):
        raise ValueError('Invalid routing population')
    channels = {}
    for i, name in enumerate(('police', 'medical', 'fire')):
        tp = int(((y[:, i] == 1) & (p[:, i] == 1)).sum())
        fp = int(((y[:, i] == 0) & (p[:, i] == 1)).sum())
        fn = int(((y[:, i] == 1) & (p[:, i] == 0)).sum())
        tn = len(y) - tp - fp - fn
        channels[name] = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'support': tp + fn,
                          'precision': tp / (tp + fp) if tp + fp else 0.,
                          'recall': tp / (tp + fn) if tp + fn else 0.,
                          'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}
    vector_counts = Counter(''.join(map(str, row)) for row in p.tolist())
    return {'videos': len(y), 'channels': channels,
            'macro_f1': float(np.mean([v['f1'] for v in channels.values()])),
            'hamming_loss': float((p != y).mean()), 'exact_match': float((p == y).all(axis=1).mean()),
            'predicted_vector_counts': {f'{i:03b}': vector_counts[f'{i:03b}'] for i in range(8)}}


def binary_metrics(rows):
    y = np.asarray([r['binary_true'] for r in rows])
    p = np.asarray([r['binary_pred'] for r in rows])
    tp, fp, fn, tn = (int(m.sum()) for m in ((y == 1) & (p == 1), (y == 0) & (p == 1), (y == 1) & (p == 0), (y == 0) & (p == 0)))
    return {'videos': len(rows), 'auroc': float(roc_auc_score(y, [r['binary_score'] for r in rows])),
            'accuracy': float((p == y).mean()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else 0., 'recall': tp / (tp + fn) if tp + fn else 0.,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def temporal_summary(metrics):
    if not metrics:
        return {'videos': 0, 'null_reason': 'no eligible observations'}
    result = {'videos': len(metrics)}
    for key in ('ap', 'top_hit', 'iou', 'zero_iou'):
        values = [m[key] for m in metrics if m[key] is not None]
        result[key + '_mean'] = float(np.mean(values)) if values else None
    values = [m['iou'] for m in metrics]
    result['iou_median'] = float(np.median(values))
    result['zero_iou_count'] = sum(x == 0 for x in values)
    result['iou_at_least'] = {str(t): float(np.mean(np.asarray(values) >= t)) for t in (.1, .3, .5, .7)}
    return result


def checkpoint_summary(records):
    if len(records) != 290 or len({r['video_id'] for r in records}) != 290 or any(r['failures'] or r['status'] == 'technical_failure' for r in records):
        raise ValueError('Official aggregate requires all 290 videos without technical failures')
    eligible = [r for r in records if r['annotation_status'] == 'anomaly']
    normal = [r for r in records if r['annotation_status'] == 'normal']
    routing = [r['routing'] for r in records]
    conditional = [r for r in routing if r['binary_true'] == 1]
    assert (len(eligible), len(normal), len(conditional)) == (135, 150, 140)
    priorities = Counter(r['cad_priority'] for r in routing)
    result = {'temporal': temporal_summary([r['temporal_metrics']['primary'] for r in eligible]),
              'category_temporal': {}, 'binary': binary_metrics(routing), 'routing': {},
              'cad_priority_counts': {k: priorities[k] for k in ('CLEAR', 'P1-EMERGENCY', 'P1-MEDICAL', 'P2-POLICE', 'P3-REVIEW')}}
    for group in sorted({r['metadata']['group'] for r in records}):
        subset = [r['temporal_metrics']['primary'] for r in eligible if r['metadata']['group'] == group]
        result['category_temporal'][group] = temporal_summary(subset)
    display = [r['temporal_metrics']['display_gated_iou'] for r in eligible]
    result['display_gated'] = {'videos': len(display), 'iou_mean': float(np.mean(display)),
                               'iou_median': float(np.median(display)), 'zero_iou_count': sum(x == 0 for x in display),
                               'zero_iou_frequency': float(np.mean(np.asarray(display) == 0))}
    result['category_display_gated'] = {}
    for group in sorted({r['metadata']['group'] for r in records}):
        vals = [r['temporal_metrics']['display_gated_iou'] for r in eligible if r['metadata']['group'] == group]
        result['category_display_gated'][group] = ({'videos': len(vals), 'iou_mean': float(np.mean(vals)),
            'iou_median': float(np.median(vals)), 'zero_iou_count': vals.count(0), 'zero_iou_frequency': vals.count(0) / len(vals)}
            if vals else {'videos': 0, 'null_reason': 'no eligible observations'})
    result['normal_temporal'] = {}
    for scope in ('primary_ungated', 'display_gated'):
        lengths = [interval_length(r['intervals'][scope]) for r in normal]
        fractions = [n / r['metadata']['src_frame_count'] for n, r in zip(lengths, normal)]
        result['normal_temporal'][scope] = {'videos': len(normal), 'support_fraction_mean': float(np.mean(fractions)),
                                           'any_interval_frequency': float(np.mean(np.asarray(lengths) > 0))}
    for scope in ('primary', 'demo'):
        result['routing'][scope] = {}
        for name, subset, field in [('raw', routing, 'raw_' + scope), ('gated', routing, 'gated_' + scope),
                                     ('conditional', conditional, 'raw_' + scope)]:
            result['routing'][scope][name] = routing_metrics([r['dispatch_target'] for r in subset], [r[field] for r in subset])
    result['routing_baselines'] = {}
    for name, subset in [('all', routing), ('conditional', conditional)]:
        targets = [r['dispatch_target'] for r in subset]
        result['routing_baselines'][name] = {
            'all_services': routing_metrics(targets, [[1, 1, 1] for r in subset]),
            'binary_gated_all_services': routing_metrics(targets, [[r['binary_pred']] * 3 for r in subset])}
    return result


def three_checkpoint_statistics(values):
    """Apply mean/sample SD to corresponding numeric leaves, retaining nulls."""
    assert len(values) == 3
    if all(isinstance(v, dict) for v in values):
        keys = sorted(set().union(*(v.keys() for v in values)))
        return {k: three_checkpoint_statistics([v.get(k) for v in values]) for k in keys}
    if all(isinstance(v, (int, float)) for v in values):
        return {'values': values, 'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1)), 'n': 3}
    return {'values': values, 'mean': None, 'sample_sd': None}
