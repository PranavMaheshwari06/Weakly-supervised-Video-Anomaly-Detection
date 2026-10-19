"""Immutable, bounded extraction for the frozen temporal_system_v1 protocol.

Run from the repository root with the repository venv. No training or MIL runs.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import cv2
import torch
import torch.nn.functional as F

from stage1b_finetune.src.clip_dataset import decode_clip_frames, transform_clip, make_clip_key
from stage1b_finetune.src import lora_backbone
from stage2_class.src.eval.temporal_geometry import clip_support
from stage2_class.src.utils.provenance import sha256_file, feature_store_identity

PHASE4 = Path('final_thesis_execution/phase_04_temporal_design_compute_preflight')
PROTOCOL_SHA = '6a3757b21650f995a216e4b88cfc916c2ff875a39de478019268528a18d9e5a0'
INDEX = Path('data/ucf_crime/artifacts/dman_v21/clip_index/videos.jsonl')
SPARSE = Path('data/ucf_crime/artifacts/dman_v21/features_lora')
PROTOTYPES = Path('artifacts/stage2/memory_lora/global_prototypes_k100.pt')
LORA = Path('artifacts/stage1b/checkpoints/lora_best.pt')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def save_tensor(path, data):
    with Path(path).open('xb') as f:
        torch.save(data, f)


def teacher_scores(features, prototypes):
    x = F.normalize(features.float(), p=2, dim=1, eps=1e-12)
    p = prototypes.float()
    return (x.square().sum(1, keepdim=True) + p.square().sum(1)[None] - 2 * (x @ p.T)).min(1).values


def extract_batch(model, pairs, index):
    tensors = []
    for vid, start in pairs:
        meta = index[vid]
        clip_support(start, meta)
        frames = decode_clip_frames(str(Path('data/ucf_crime/raw') / vid), start, 16, meta['keep_every_n'])
        if frames is None:
            raise RuntimeError(f'decode_failure: {vid}__s{start:06d}')
        tensors.append(transform_clip(frames))
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        features = model.extract_features(torch.stack(tensors).cuda()).half().cpu()
    if features.shape != (len(pairs), 768) or not torch.isfinite(features).all():
        raise RuntimeError('shape_mismatch or nonfinite backbone features')
    return features


def replay(model, index, prototypes):
    reference = read_json(PHASE4 / 'loader_teacher_audit.json')
    state = model.backbone.state_dict()
    adapters = torch.load(LORA, map_location='cpu', weights_only=False)['lora_state_dict']
    assert len(adapters) == 48
    for key, value in adapters.items():
        assert torch.equal(state[key].cpu(), value), key
    biases = [v for k, v in state.items() if '.attention.attention.' in k and k.endswith('.bias')]
    assert len(biases) == 36 and all(torch.count_nonzero(v) == 0 for v in biases)
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        for part in (key.encode(), str(value.dtype).encode(), str(tuple(value.shape)).encode(), value.numpy().tobytes()):
            digest.update(part)
    assert digest.hexdigest() == reference['loaded_backbone_state_sha256'], 'Loaded backbone state mismatch'
    selected = read_json(PHASE4 / 'backbone_replay.json')['results']['current_loader']
    wanted = {make_clip_key(r['video_id'], r['start']) for r in selected}
    expected = {}
    old = torch.load('artifacts/stage2/teacher/teacher_scores_val.pt', map_location='cpu', weights_only=False)
    old_scores = dict(zip(old['clip_keys'], old['scores'].float().tolist()))
    differences = []
    for path in sorted(SPARSE.glob('global_val_shard_*.pt')):
        shard = torch.load(path, map_location='cpu', weights_only=False)
        scores = teacher_scores(shard['global'], prototypes)
        for i, (vid, start) in enumerate(zip(shard['video_ids'], shard['clip_starts'].tolist())):
            key = make_clip_key(vid, start)
            differences.append(abs(float(scores[i]) - old_scores[key]))
            if key in wanted:
                assert key not in expected
                expected[key] = shard['global'][i].float().clone()
    assert len(differences) == 17117 and max(differences) <= .001
    assert set(expected) == wanted
    rows = []
    for offset in range(0, len(selected), 4):
        batch = selected[offset:offset + 4]
        actual = extract_batch(model, [(r['video_id'], r['start']) for r in batch], index).float()
        for r, value in zip(batch, actual):
            target = expected[make_clip_key(r['video_id'], r['start'])]
            error = float((value - target).abs().max())
            cosine = float(F.cosine_similarity(value[None], target[None]))
            rows.append({'video_id': r['video_id'], 'start': r['start'], 'max_abs_error': error, 'cosine_similarity': cosine})
    result = {'adapter_keys_checked': len(adapters), 'new_attention_bias_count': len(biases),
              'new_attention_bias_max_abs': 0.0, 'loaded_backbone_state_sha256': digest.hexdigest(),
              'features': rows, 'teacher_validation_clips': len(differences),
              'teacher_max_abs_difference': max(differences),
              'teacher_mean_abs_difference': sum(differences) / len(differences),
              'teacher_validation_source_sha256': sha256_file('artifacts/stage2/teacher/teacher_scores_val.pt')}
    result['passed'] = all(r['max_abs_error'] <= .005 and r['cosine_similarity'] >= .99999 for r in rows)
    return result


def main():
    assert sha256_file(PHASE4 / 'protocol.json') == PROTOCOL_SHA
    protocol = read_json(PHASE4 / 'protocol.json')
    # Recheck every frozen identity and environment before any output creation.
    subprocess.run([sys.executable, '-m', 'utils.verify_temporal_preflight', '--require-output-absent'], check=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; require host GPU access, no CPU fallback')
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    outputs = protocol['outputs']
    attempt = Path(outputs['run_root']) / protocol['attempt_id']
    attempt.mkdir(parents=True, exist_ok=False)
    index = {r['rel_path']: r for r in map(json.loads, INDEX.read_text().splitlines())}
    ids = Path('data/ucf_crime/splits/dman_v21/test.txt').read_text().splitlines()
    context = {'video_id': None, 'checkpoint': None, 'stage': 'validation_replay', 'affected_keys': [],
               'expected_count': 18, 'observed_count': 0, 'retry_identity': None}
    start_time = time.perf_counter()
    try:
        cache = Path.home() / '.cache/huggingface/hub/models--MCG-NJU--videomae-base/snapshots' / protocol['feature_extraction']['cache_revision']
        # Pin the exact verified snapshot while retaining the frozen loader behavior.
        lora_backbone.BACKBONE_NAME = str(cache)
        packages = read_json(PHASE4 / 'annotation_environment_audit.json')['environment']['packages']
        environment = {'python': platform.python_version(), 'platform': platform.platform(),
                       'packages': {k: importlib.metadata.version(k) for k in packages},
                       'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                       'gpu': torch.cuda.get_device_name(), 'dedicated_vram_bytes': torch.cuda.get_device_properties(0).total_memory,
                       'cpu_threads': torch.get_num_threads(), 'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True)}
        write_json(attempt / 'execution.json', {'command': sys.argv, 'environment': environment,
                   'protocol_sha256': PROTOCOL_SHA, 'source_sha256': sha256_file(__file__),
                   'branch': subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                   'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   'git_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
                   'frozen_identities': {**protocol['input_hashes'], **protocol['source_hashes'], **protocol['evidence_hashes']},
                   'backbone_cache': read_json(PHASE4 / 'backbone_replay.json')['backbone_hashes']})
        model = lora_backbone.LoRAVideoMAE.load_from_checkpoint(str(LORA), device='cuda')
        prototypes = torch.load(PROTOTYPES, map_location='cpu', weights_only=False)['prototypes']
        assert prototypes.shape == (100, 768) and prototypes.dtype == torch.float16 and torch.isfinite(prototypes).all()
        record = replay(model, index, prototypes)
        write_json(attempt / 'validation_replay.json', record)
        assert record['passed'], 'Feature compatibility replay failed'
        print('PASS validation feature/teacher/state replay', flush=True)
        candidates = []
        for vid in ids:
            rec = index[vid]
            starts = rec['clip_starts']
            assert starts == sorted(set(starts)) and rec['split'] == 'test'
            for s in starts:
                clip_support(s, rec)
            candidates.append({'video_id': vid, 'clip_starts': starts, 'clip_keys': [make_clip_key(vid, s) for s in starts],
                               'metadata': {k: rec[k] for k in ('src_fps', 'keep_every_n', 'src_frame_count', 'clip_len', 'stride')}})
        assert len(candidates) == 290 and sum(len(r['clip_keys']) for r in candidates) == 69090
        write_json(attempt / 'candidate_manifest.json', candidates)
        store = Path(outputs['feature_store'])
        store.mkdir(parents=True, exist_ok=False)
        Path(outputs['teacher_root']).mkdir(parents=True, exist_ok=False)
        buffer_features, buffer_pairs, shards, videos = [], [], [], []

        def flush():
            if not buffer_pairs:
                return
            path = store / f'global_test_shard_{len(shards):03d}.pt'
            save_tensor(path, {'global': torch.stack(buffer_features),
                        'clip_keys': [make_clip_key(v, s) for v, s in buffer_pairs],
                        'video_ids': [v for v, s in buffer_pairs],
                        'clip_starts': torch.tensor([s for v, s in buffer_pairs]),
                        'meta': {'protocol_sha256': PROTOCOL_SHA, 'candidate_manifest_sha256': sha256_file(attempt / 'candidate_manifest.json')}})
            shards.append({'path': path.name, 'sha256': sha256_file(path), 'clips': len(buffer_pairs)})
            buffer_features.clear()
            buffer_pairs.clear()

        torch.cuda.reset_peak_memory_stats()
        for row in candidates:
            vid = row['video_id']
            context.update(video_id=vid, stage='extraction', expected_count=len(row['clip_keys']), observed_count=0)
            raw = Path('data/ucf_crime/raw') / vid
            before = raw.stat()
            cap = cv2.VideoCapture(str(raw))
            probe = {'opened': cap.isOpened(), 'fps': cap.get(cv2.CAP_PROP_FPS), 'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
            cap.release()
            assert probe == {'opened': True, 'fps': row['metadata']['src_fps'], 'frames': row['metadata']['src_frame_count'], 'width': 320, 'height': 240}, 'source drift'
            t0 = time.perf_counter()
            pairs = [(vid, s) for s in row['clip_starts']]
            for offset in range(0, len(pairs), 4):
                batch = pairs[offset:offset + 4]
                context['affected_keys'] = [make_clip_key(v, s) for v, s in batch]
                values = extract_batch(model, batch, index)
                for pair, feature in zip(batch, values):
                    buffer_pairs.append(pair)
                    buffer_features.append(feature.clone())
                    if len(buffer_pairs) == 512:
                        flush()
                context['observed_count'] += len(batch)
            after = raw.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'source changed during extraction'
            video = {**row, 'realized_clip_keys': row['clip_keys'], 'raw_bytes': before.st_size, 'raw_mtime_ns': before.st_mtime_ns,
                     'raw_probe': probe, 'wall_seconds': time.perf_counter() - t0}
            write_json(attempt / f'extraction_video_{len(videos):03d}.json', video)
            videos.append(video)
            print(json.dumps({'videos': len(videos), 'clips': sum(len(v['clip_keys']) for v in videos),
                              'video_id': vid, 'seconds': video['wall_seconds']}), flush=True)
        flush()
        identity = feature_store_identity(store, 'test')
        write_json(store / 'manifest.json', {'protocol_sha256': PROTOCOL_SHA, 'candidate_manifest_sha256': sha256_file(attempt / 'candidate_manifest.json'),
                   'identity': identity, 'shards': shards, 'clips': 69090, 'videos': 290,
                   'validation_replay_sha256': sha256_file(attempt / 'validation_replay.json'),
                   'execution_sha256': sha256_file(attempt / 'execution.json')})
        context.update(video_id=None, stage='teacher', affected_keys=[], expected_count=69090, observed_count=0)
        keys, scores = [], []
        for path in sorted(store.glob('global_test_shard_*.pt')):
            shard = torch.load(path, map_location='cpu', weights_only=False)
            keys.extend(shard['clip_keys'])
            scores.append(teacher_scores(shard['global'], prototypes))
        scores = torch.cat(scores)
        assert len(keys) == len(set(keys)) == len(scores) == 69090 and torch.isfinite(scores).all()
        teacher_path = Path(outputs['teacher_root']) / 'teacher_scores_test.pt'
        save_tensor(teacher_path, {'clip_keys': keys, 'scores': scores, 'meta': {'prototype_sha256': sha256_file(PROTOTYPES),
                    'feature_store_identity': identity['identity_sha256'], 'protocol_sha256': PROTOCOL_SHA}})
        write_json(attempt / 'extraction_complete.json', {'status': 'complete', 'clips': len(keys), 'videos': len(videos),
                   'feature_store_identity': identity, 'teacher_sha256': sha256_file(teacher_path),
                   'wall_seconds': time.perf_counter() - start_time, 'extraction_video_seconds': sum(v['wall_seconds'] for v in videos),
                   'max_allocated_bytes': torch.cuda.max_memory_allocated(), 'max_reserved_bytes': torch.cuda.max_memory_reserved(), 'failures': []})
        print('PASS extraction and teacher: 69090 clips / 290 videos', flush=True)
    except BaseException as exc:
        code = 'OOM' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'decode_failure' if 'decode_failure' in str(exc) else 'execution_failure'
        write_json(attempt / 'failure.json', {**context, 'code': code, 'message': repr(exc), 'wall_seconds': time.perf_counter() - start_time})
        raise


if __name__ == '__main__':
    main()
