"""Source-frame geometry for Phase 4/5; intervals are zero-based, half-open.

The index describes integer decimation, not interpolation to exactly 15 FPS.
Clip support includes each sampled frame's decimation cell, clipped to EOF.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import gaussian_filter1d


def validate_metadata(meta: dict) -> None:
    fps = float(meta['src_fps'])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid source FPS')
    for name in ('keep_every_n', 'src_frame_count', 'clip_len', 'stride'):
        if isinstance(meta[name], bool) or int(meta[name]) != meta[name] or meta[name] <= 0:
            raise ValueError(f'Invalid {name}')
    if meta['keep_every_n'] != max(1, round(fps / 15.0)):
        raise ValueError('Index decimation disagrees with decoder')


def clip_support(start: int, meta: dict) -> tuple[int, int]:
    validate_metadata(meta)
    if isinstance(start, bool) or int(start) != start or start < 0:
        raise ValueError('Invalid clip start')
    n, length, total = meta['keep_every_n'], meta['clip_len'], meta['src_frame_count']
    if (start + length - 1) * n >= total:
        raise ValueError('Clip requires samples beyond EOF')
    return int(start * n), min(int((start + length) * n), int(total))


def frame_to_seconds(frame: int, meta: dict) -> float:
    validate_metadata(meta)
    if not 0 <= frame <= meta['src_frame_count']:
        raise ValueError('Frame boundary outside video')
    return frame / float(meta['src_fps'])


def merge_intervals(intervals) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(intervals):
        if start < 0 or end <= start:
            raise ValueError('Invalid half-open interval')
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def interval_length(intervals) -> int:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersection_length(first, second) -> int:
    a, b = merge_intervals(first), merge_intervals(second)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def annotation_intervals(pairs, total: int) -> list[tuple[int, int]]:
    """Frozen convention: positive annotation frame numbers are 1-based inclusive.

    Missing annotations must be represented separately, never as empty pairs.
    Out-of-range annotations fail rather than silently clipping ground truth.
    """
    result = []
    for start, end in pairs:
        if (start, end) == (-1, -1):
            continue
        if int(start) != start or int(end) != end or not 1 <= start <= end <= total:
            raise ValueError(f'Invalid annotation bounds: {(start, end)} / {total}')
        result.append((int(start - 1), int(end)))
    return merge_intervals(result)


def smooth_attention(attention, starts, meta: dict) -> np.ndarray:
    """Sigma=2 within stride-contiguous runs; normalize once per video.

    A missing candidate breaks smoothing. Constant attention maps to zeros.
    """
    validate_metadata(meta)
    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(starts) or not np.isfinite(values).all():
        raise ValueError('Invalid attention')
    if list(starts) != sorted(set(starts)):
        raise ValueError('Clip starts must be unique and ordered')
    for start in starts:
        clip_support(start, meta)
    if not len(values):
        return values
    breaks = [0] + [i for i in range(1, len(starts))
                    if starts[i] - starts[i-1] != meta['stride']] + [len(starts)]
    smoothed = np.empty_like(values)
    for begin, end in zip(breaks[:-1], breaks[1:]):
        smoothed[begin:end] = gaussian_filter1d(values[begin:end], sigma=2, mode='reflect')
    low, high = smoothed.min(), smoothed.max()
    return (smoothed - low) / (high - low) if high > low else np.zeros_like(values)


def attention_intervals(attention, starts, meta: dict, threshold: float = 0.5):
    if not 0 < threshold <= 1:
        raise ValueError('Threshold must be in (0, 1]')
    scores = smooth_attention(attention, starts, meta)
    return merge_intervals([clip_support(s, meta) for s, score in zip(starts, scores)
                            if score >= threshold])
