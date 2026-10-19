"""Independent Phase-4 integrity/coverage verification; no model forward pass."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import torch

from stage2_class.src.utils.provenance import feature_store_identity

ROOT = Path('reproducibility/temporal_protocol')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--require-output-absent',action='store_true')
    args=parser.parse_args()
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert protocol['protocol_id']=='temporal_system_v1'
    assert protocol['metrics']['random_baseline'] == {
        'rng': 'numpy.random.Generator',
        'bit_generator': 'PCG64',
        'seed_derivation': {
            'hash': 'sha256', 'input': 'temporal_system_v1:{video_id}',
            'encoding': 'utf-8', 'digest_byte_slice': [0, 8],
            'byte_order': 'big', 'signed': False,
        },
        'replicates_per_video': 1,
        'draw': 'rng.random(M, dtype=numpy.float64), Uniform[0,1)',
        'clip_order': ['clip_start', 'clip_key'],
        'normalization': 'divide the M draws by their sum',
        'reuse_across_checkpoints': True,
        'checkpoint_independent': True,
        'aggregation': 'one seed-independent baseline value per video; do not count checkpoint reuse as independent replicates or report a three-seed baseline SD',
        'selection': 'no best-of-seeds, favorable-draw selection, or additional random replicates',
    }, 'Random-baseline RNG/replicate rule differs from the frozen protocol'
    hashes={**protocol['input_hashes'],**protocol['source_hashes'],**protocol['evidence_hashes']}
    for path,expected in hashes.items():
        # The frozen protocol records its companion files under their original folder.
        path=path.replace('final_thesis_execution/phase_04_temporal_design_compute_preflight',str(ROOT))
        assert sha(path)==expected, f'Changed frozen input: {path}'
    assert [x['training_seed'] for x in protocol['checkpoints']]==[1337,2027,3407]
    for ckpt in protocol['checkpoints']:
        d=torch.load(ckpt['checkpoint'],map_location='cpu',weights_only=False)
        assert d['meta']['selected_epoch']==ckpt['selected_epoch']
        assert d['meta']['run_tag']==ckpt['run_tag']
        assert d['meta']['configuration_identity']==ckpt['resolved_config_sha256']
        assert d['meta']['input_dim']==769 and d['meta']['num_dispatch']==3
    identity=feature_store_identity('data/ucf_crime/artifacts/dman_v21/features_lora','test')
    assert identity['identity_sha256']==protocol['sparse_feature_identity']['identity_sha256']
    audit=json.loads((ROOT/'coverage_audit.json').read_text())
    index={r['rel_path']:r for r in map(json.loads,Path('data/ucf_crime/artifacts/dman_v21/clip_index/videos.jsonl').read_text().splitlines())}
    sparse=collections.defaultdict(list)
    for path in sorted(Path('data/ucf_crime/artifacts/dman_v21/features_lora').glob('global_test_shard_*.pt')):
        d=torch.load(path,map_location='cpu',weights_only=False)
        for vid,start in zip(d['video_ids'],d['clip_starts'].tolist()): sparse[vid].append(start)
    bags=json.loads(Path('artifacts/stage2/bag_manifests/dispatch_lora_fixed_test.json').read_text())['bags']
    test_ids=Path('data/ucf_crime/splits/dman_v21/test.txt').read_text().splitlines()
    assert [r['video_id'] for r in audit['videos']]==test_ids
    for row in audit['videos']:
        vid=row['video_id']; meta=index[vid]; total=meta['src_frame_count']; n=meta['keep_every_n']
        assert row['raw_probe']['opened'] and row['raw_probe']['fps']==meta['src_fps'] and row['raw_probe']['frames']==total
        for label,starts in [('fixed_bag',[int(k.rsplit('__s',1)[1]) for k in bags[vid]]),('sparse',sparse[vid]),('dense_candidate',meta['clip_starts'])]:
            # Independent boolean-mask calculation, not temporal_geometry helpers.
            mask=np.zeros(total,dtype=bool)
            for start in starts:
                assert (start+15)*n<total
                mask[start*n:min((start+16)*n,total)]=True
            measured=row['coverage'][label]
            assert measured['clips']==len(starts)
            assert measured['unique_clips']==len(set(starts))
            assert measured['covered_source_frames']==int(mask.sum())
            assert measured['fraction']==float(mask.mean())
            if row['gt_intervals_half_open']:
                gt=np.zeros(total,dtype=bool)
                for s,e in row['gt_intervals_half_open']: gt[s:e]=True
                overlap=int((gt & mask).sum())
                assert measured['annotated_frames_covered']==overlap
                assert measured['annotation_coverage']==overlap/int(gt.sum())
    for label in ('fixed_bag','sparse','dense_candidate'):
        values=[r['coverage'][label] for r in audit['videos']]
        actual=audit['summary'][label]
        assert actual['clips']==sum(v['clips'] for v in values)
        assert math.isclose(actual['video_mean_coverage'],np.mean([v['fraction'] for v in values]),abs_tol=1e-15)
        assert actual['frame_weighted_coverage']==sum(v['covered_source_frames'] for v in values)/sum(r['metadata']['src_frame_count'] for r in audit['videos'])
    annotations=json.loads((ROOT/'annotation_environment_audit.json').read_text())
    assert annotations['matlab_text_raw_conflicts']==[]
    assert all(r['metadata_frame_count']==r['decoded_frame_count'] for r in annotations['fully_decoded_boundary_conflicts'])
    assert audit['annotation_status_counts']['anomaly']==135 and audit['annotation_status_counts']['normal']==150
    assert len(protocol['evaluation_population']['temporal_excluded'])==5
    benchmark=json.loads((ROOT/'benchmark_results.json').read_text())
    sample=json.loads((ROOT/'benchmark_sample.json').read_text())
    assert benchmark['sample_sha256']==sha(ROOT/'benchmark_sample.json')
    val_ids=set(Path('data/ucf_crime/splits/dman_v21/val.txt').read_text().splitlines())
    assert all(r['video_id'] in val_ids and r['video_id'] not in test_ids for r in sample['samples'])
    assert len(benchmark['rows'])==12 and benchmark['total_measured_clips']==594
    assert benchmark['total_measured_clips']==sum(r['clips'] for r in benchmark['rows'])
    for row in benchmark['rows']:
        planned=next(s for s in sample['samples'] if s['video_id']==row['video_id'])
        assert row['clips']==len(planned['starts'])
        assert row['wall_seconds']>0 and 0<row['max_allocated_bytes']<=row['max_reserved_bytes']<row['cuda_total_bytes']
    projected=69090*sum(r['wall_seconds'] for r in benchmark['rows'])/594
    assert math.isclose(projected,benchmark['projection']['weighted_seconds'],abs_tol=1e-9)
    replay=json.loads((ROOT/'backbone_replay.json').read_text())
    cache=Path.home()/'.cache/huggingface/hub/models--MCG-NJU--videomae-base/snapshots'/replay['backbone_cache_revision']
    for name,expected in replay['backbone_hashes'].items():
        assert sha(cache/name)==expected
    for name,expected in annotations['environment']['packages'].items():
        assert importlib.metadata.version(name)==expected, f'Environment drift: {name}'
    for row in replay['results']['current_loader']:
        assert row['max_abs_error']<=.005 and row['cosine_similarity']>=.99999
    teacher=json.loads((ROOT/'loader_teacher_audit.json').read_text())
    assert teacher['adapter_keys_checked']==48 and teacher['new_attention_bias_count']==36 and teacher['new_attention_bias_max_abs']==0
    assert teacher['teacher_validation_clips']==17117 and teacher['teacher_max_abs_difference']<=.001
    schema=json.loads((ROOT/'PER_VIDEO_SCHEMA.json').read_text())
    assert set(schema['required'])<=set(schema['properties'])
    assert schema['properties']['protocol_id']['const']==protocol['protocol_id']
    if args.require_output_absent:
        assert all(not Path(p).exists() for p in protocol['outputs'].values())
    print(f'PASS: {len(hashes)} frozen hashes; 3 checkpoint identities; 290 x 3 coverage records; 135 eligible temporal anomalies; 594 benchmark clips; feature/teacher compatibility; output schema structure')


if __name__=='__main__': main()
