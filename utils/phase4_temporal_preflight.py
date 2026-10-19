"""Measure coverage and freeze a metadata-selected benchmark; no MIL inference."""
from __future__ import annotations

import collections
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat

from stage2_class.src.eval.temporal_geometry import (
    annotation_intervals, clip_support, interval_length, intersection_length, merge_intervals,
)
from stage2_class.src.utils.provenance import sha256_file, feature_store_identity

OUT = Path('reproducibility/temporal_protocol')
INDEX = Path('data/ucf_crime/artifacts/dman_v21/clip_index/videos.jsonl')
SPLITS = Path('data/ucf_crime/splits/dman_v21')
ANNOTATIONS = Path('data/ucf_crime/annotations/Temporal_Anomaly_Annotation_For_Testing_Videos')
STORE = Path('data/ucf_crime/artifacts/dman_v21/features_lora')


def write_new(path, data):
    with path.open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def main():
    index = {r['rel_path']: r for r in map(json.loads, INDEX.read_text().splitlines())}
    ids = (SPLITS / 'test.txt').read_text().splitlines()
    ann_path = ANNOTATIONS / 'Txt_formate/Temporal_Anomaly_Annotation.txt'
    annotations = {}
    for line in ann_path.read_text().splitlines():
        if not line.strip():
            continue
        name, category, *bounds = line.split()
        assert name not in annotations and len(bounds) == 4
        annotations[name] = (category, list(zip(map(int, bounds[::2]), map(int, bounds[1::2]))))
    sparse = collections.defaultdict(list)
    for path in sorted(STORE.glob('global_test_shard_*.pt')):
        shard = torch.load(path, map_location='cpu', weights_only=False)
        for vid, start in zip(shard['video_ids'], shard['clip_starts'].tolist()):
            sparse[vid].append(start)
    fixed_path = Path('artifacts/stage2/bag_manifests/dispatch_lora_fixed_test.json')
    bags = json.loads(fixed_path.read_text())['bags']
    rows, mat_conflicts = [], []
    for vid in ids:
        rec = index[vid]
        name = Path(vid).name
        gt, reason = None, None
        if name not in annotations:
            reason = 'missing_annotation'
        else:
            category, pairs = annotations[name]
            try:
                gt = annotation_intervals(pairs, rec['src_frame_count'])
                if bool(gt) != bool(rec['label_bin']):
                    reason = 'annotation_label_conflict'
                mat_path = ANNOTATIONS / 'Matlab_formate' / (Path(name).stem + '.mat')
                mat = loadmat(mat_path, simplify_cells=True)['Annotation_file']
                mat_pairs = np.asarray(mat['Anno']).reshape(-1, 2).tolist()
                if annotation_intervals(mat_pairs, rec['src_frame_count']) != gt:
                    mat_conflicts.append(vid)
            except (ValueError, FileNotFoundError) as exc:
                reason = str(exc)
        raw_path = Path('data/ucf_crime/raw') / vid
        cap = cv2.VideoCapture(str(raw_path))
        raw = {'opened': cap.isOpened(), 'fps': cap.get(cv2.CAP_PROP_FPS),
               'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
               'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        cap.release()
        row = {'video_id': vid, 'label_bin': rec['label_bin'], 'group': rec['group'],
               'metadata': {k: rec[k] for k in ('src_fps','src_frame_count','keep_every_n','clip_len','stride')},
               'raw_probe': raw, 'annotation_status': reason or ('anomaly' if gt else 'normal'),
               'gt_intervals_half_open': gt, 'coverage': {}}
        assert raw['opened'] and raw['fps'] == rec['src_fps'] and raw['frames'] == rec['src_frame_count'], vid
        assert len(sparse[vid]) == len(set(sparse[vid])) and set(sparse[vid]) <= set(rec['clip_starts'])
        for label, starts in [('fixed_bag', [int(k.rsplit('__s',1)[1]) for k in bags[vid]]),
                              ('sparse', sparse[vid]), ('dense_candidate', rec['clip_starts'])]:
            support = merge_intervals([clip_support(s, rec) for s in sorted(set(starts))])
            covered = interval_length(support)
            gaps = [support[0][0]] + [b[0]-a[1] for a,b in zip(support[:-1],support[1:])] + [rec['src_frame_count']-support[-1][1]]
            row['coverage'][label] = {'clips':len(starts),'unique_clips':len(set(starts)),
                'covered_source_frames':covered,'fraction':covered/rec['src_frame_count'],
                'max_uncovered_gap_frames':max(gaps),
                'annotated_frames_covered': intersection_length(support, gt) if gt else None,
                'annotation_coverage': intersection_length(support, gt)/interval_length(gt) if gt else None}
        rows.append(row)
    summary = {}
    for label in ('fixed_bag','sparse','dense_candidate'):
        vals = [r['coverage'][label]['fraction'] for r in rows]
        ann = [r['coverage'][label]['annotation_coverage'] for r in rows if r['coverage'][label]['annotation_coverage'] is not None]
        summary[label] = {'clips':sum(r['coverage'][label]['clips'] for r in rows),
            'video_mean_coverage':float(np.mean(vals)), 'video_median_coverage':float(np.median(vals)),
            'min_coverage':min(vals), 'frame_weighted_coverage':sum(r['coverage'][label]['covered_source_frames'] for r in rows)/sum(r['metadata']['src_frame_count'] for r in rows),
            'anomaly_mean_annotation_coverage':float(np.mean(ann)), 'anomalies_with_zero_annotation_coverage':sum(x==0 for x in ann)}
    important = [INDEX, ann_path, ANNOTATIONS/'ReadMe.docx', fixed_path, SPLITS/'test.txt', SPLITS/'val.txt',
                 Path('artifacts/stage1b/checkpoints/lora_best.pt'),
                 Path('artifacts/stage2/memory_lora/global_prototypes_k100.pt'),
                 Path('artifacts/stage2/teacher/teacher_scores_test.pt'),
                 Path('artifacts/stage2/evidence/teacher_ablation_v1/summary.json')]
    for seed in (1337,2027,3407):
        root=Path(f'artifacts/stage2/runs/dispatch_repro_s{seed}')
        important += [root/'checkpoints/mil_best.pt',root/'resolved_config.yaml',root/'run_manifest.json']
    payload = {'kind':'coverage_and_annotation_audit_not_model_evaluation','input_hashes':{str(p):sha256_file(p) for p in important},
        'feature_identity':feature_store_identity(str(STORE),'test'), 'annotation_count':len(annotations),
        'annotation_status_counts':dict(collections.Counter(r['annotation_status'] for r in rows)),
        'matlab_text_conflicts':mat_conflicts,'summary':summary,'videos':rows}
    write_new(OUT/'coverage_audit.json',payload)
    # Select 6 validation videos by duration: min/median/max within each binary class.
    # No temporal annotation or model output is consulted for sample selection.
    sample=[]
    val_ids=(SPLITS/'val.txt').read_text().splitlines()
    for label in (0,1):
        ordered=sorted((index[v] for v in val_ids if index[v]['label_bin']==label),key=lambda r:(r['src_frame_count']/r['src_fps'],r['rel_path']))
        for stratum,i in [('short',0),('median',len(ordered)//2),('long',len(ordered)-1)]:
            rec=ordered[i]; count=min(64,len(rec['clip_starts']))
            # Contiguous 16-clip windows at the beginning, middle and end plus one intermediate window.
            offsets=np.linspace(0,max(0,len(rec['clip_starts'])-16),4,dtype=int)
            starts=sorted({s for off in offsets for s in rec['clip_starts'][off:off+16]})
            sample.append({'video_id':rec['rel_path'],'label_bin':label,'duration_stratum':stratum,
                           'duration_seconds':rec['src_frame_count']/rec['src_fps'], 'metadata':{k:rec[k] for k in ('src_fps','src_frame_count','keep_every_n','clip_len','stride')},
                           'candidate_clips':len(rec['clip_starts']),'starts':starts})
    write_new(OUT/'benchmark_sample.json',{'selection':'validation; short/median/long within normal/anomaly; four spread contiguous windows of up to 16 clips; no model or temporal labels',
        'batch_size':4,'repeats':2,'warmup_batches':2,'samples':sample})
    print(json.dumps({'coverage':summary,'annotation_status_counts':payload['annotation_status_counts'],'matlab_conflicts':mat_conflicts,
                      'benchmark':[(r['video_id'],len(r['starts'])) for r in sample]},indent=2))


if __name__ == '__main__':
    main()
