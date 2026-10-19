"""Technical continuation of interrupted attempt_001; reuse every persisted key.

This creates attempt_002, preserves attempt_001 byte-for-byte, and exclusively
adds missing shard files. It never reruns a clip already present on disk.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch

from utils.extract_temporal_system import (
    PHASE4, PROTOCOL_SHA, INDEX, LORA, PROTOTYPES, lora_backbone,
    read_json, write_json, save_tensor, extract_batch, replay, teacher_scores,
    make_clip_key, sha256_file, feature_store_identity,
)


def inspect_prefix(store, candidates):
    expected = [k for row in candidates for k in row['clip_keys']]
    keys, shards = [], []
    for i, path in enumerate(sorted(store.glob('global_test_shard_*.pt'))):
        assert path.name == f'global_test_shard_{i:03d}.pt'
        obj = torch.load(path, map_location='cpu', weights_only=False)
        assert obj['global'].dtype == torch.float16 and obj['global'].shape == (len(obj['clip_keys']), 768)
        assert 0 < len(obj['clip_keys']) <= 512 and torch.isfinite(obj['global']).all()
        assert len(obj['video_ids']) == len(obj['clip_starts']) == len(obj['clip_keys'])
        assert all(k == make_clip_key(v, s) for k, v, s in zip(obj['clip_keys'], obj['video_ids'], obj['clip_starts'].tolist()))
        keys.extend(obj['clip_keys'])
        shards.append({'path': path.name, 'sha256': sha256_file(path), 'clips': len(obj['clip_keys'])})
    assert keys == expected[:len(keys)] and len(keys) == len(set(keys))
    return keys, shards


def main():
    assert sha256_file(PHASE4 / 'protocol.json') == PROTOCOL_SHA
    subprocess.run([sys.executable, '-m', 'utils.verify_temporal_preflight'], check=True)
    protocol = read_json(PHASE4 / 'protocol.json')
    outputs = protocol['outputs']
    original = Path(outputs['run_root']) / 'attempt_001'
    attempt = Path(outputs['run_root']) / 'attempt_002'
    store = Path(outputs['feature_store'])
    assert not (original / 'extraction_complete.json').exists()
    assert not (store / 'manifest.json').exists()
    assert not any(Path(outputs['teacher_root']).iterdir())
    if not torch.cuda.is_available():
        raise RuntimeError('Host CUDA access required')
    # Guard against duplicating an active original extraction.
    processes = subprocess.run(['pgrep', '-af', '^[^ ]*python[^ ]* -m utils.extract_temporal_system'], capture_output=True, text=True)
    assert processes.returncode == 1, 'Original extraction may still be active'
    candidate_path = original / 'candidate_manifest.json'
    candidates = read_json(candidate_path)
    keys, shards = inspect_prefix(store, candidates)
    assert len(keys) == 11264 and len(shards) == 22, 'Unexpected interruption boundary'
    prior = {str(p): sha256_file(p) for p in original.rglob('*') if p.is_file()}
    previous_execution = read_json(original / 'execution.json')
    assert previous_execution['source_sha256'] == sha256_file('utils/extract_temporal_system.py')
    attempt.mkdir(exist_ok=False)
    context = {'video_id': None, 'checkpoint': None, 'stage': 'recovery_replay', 'affected_keys': [],
               'expected_count': 69090, 'observed_count': len(keys), 'retry_identity': 'attempt_002'}
    started = time.perf_counter()
    try:
        write_json(attempt / 'recovery_plan.json', {'reason': 'User-reported session interruption; host process absent; attempt_001 did not finalize',
            'previous_attempt': str(original), 'previous_attempt_files': prior, 'retained_shards': shards.copy(),
            'retained_clips': len(keys), 'remaining_clips': 69090 - len(keys), 'first_missing_key':
            [k for r in candidates for k in r['clip_keys']][len(keys)], 'candidate_manifest_path': str(candidate_path),
            'candidate_manifest_sha256': sha256_file(candidate_path), 'protocol_sha256': PROTOCOL_SHA,
            'failure': {**context, 'stage': 'extraction', 'code': 'session_interrupted',
                        'message': 'Original process absent; only validated persisted prefix can be reused'},
            'policy': 'Preserve every existing file; new attempt; append exclusive new shards; extract only missing keys',
            'timing_limitation': 'Unrecorded work in interrupted partial video and lost in-memory features has no recoverable timing; no full uninterrupted runtime claim'})
        execution = {**previous_execution, 'command': sys.argv, 'attempt_id': 'attempt_002',
                     'resumed_from': 'attempt_001', 'recovery_source_sha256': sha256_file(__file__),
                     'environment': {**previous_execution['environment'], 'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True)}}
        write_json(attempt / 'execution.json', execution)
        torch.set_num_threads(4); torch.manual_seed(1337)
        cache = Path.home() / '.cache/huggingface/hub/models--MCG-NJU--videomae-base/snapshots' / protocol['feature_extraction']['cache_revision']
        lora_backbone.BACKBONE_NAME = str(cache)
        index = {r['rel_path']: r for r in map(json.loads, INDEX.read_text().splitlines())}
        prototypes = torch.load(PROTOTYPES, map_location='cpu', weights_only=False)['prototypes']
        model = lora_backbone.LoRAVideoMAE.load_from_checkpoint(str(LORA), device='cuda')
        validation = replay(model, index, prototypes)
        write_json(attempt / 'validation_replay.json', validation)
        assert validation['passed']
        print('PASS recovery replay; reusing 11264 persisted clips without re-extraction', flush=True)
        retained = set(keys)
        buffer_pairs, buffer_features, video_paths = [], [], []

        def flush():
            if not buffer_pairs:
                return
            path = store / f'global_test_shard_{len(shards):03d}.pt'
            save_tensor(path, {'global': torch.stack(buffer_features), 'clip_keys': [make_clip_key(v, s) for v, s in buffer_pairs],
                'video_ids': [v for v, s in buffer_pairs], 'clip_starts': torch.tensor([s for v, s in buffer_pairs]),
                'meta': {'protocol_sha256': PROTOCOL_SHA, 'candidate_manifest_sha256': sha256_file(candidate_path), 'attempt_id': 'attempt_002'}})
            shards.append({'path': path.name, 'sha256': sha256_file(path), 'clips': len(buffer_pairs)})
            buffer_pairs.clear(); buffer_features.clear()

        torch.cuda.reset_peak_memory_stats()
        new_count = 0
        for number, row in enumerate(candidates):
            vid = row['video_id']
            pending = [(vid, s) for s, k in zip(row['clip_starts'], row['clip_keys']) if k not in retained]
            old_record = original / f'extraction_video_{number:03d}.json'
            if not pending and old_record.exists():
                old = read_json(old_record)
                raw_stat = (Path('data/ucf_crime/raw') / vid).stat()
                assert old['realized_clip_keys'] == row['clip_keys']
                assert (old['raw_bytes'], old['raw_mtime_ns']) == (raw_stat.st_size, raw_stat.st_mtime_ns)
                video_paths.append(str(old_record))
                continue
            context.update(video_id=vid, stage='extraction', expected_count=len(row['clip_keys']), observed_count=len(row['clip_keys']) - len(pending))
            raw = Path('data/ucf_crime/raw') / vid
            before = raw.stat(); cap = cv2.VideoCapture(str(raw))
            probe = {'opened': cap.isOpened(), 'fps': cap.get(cv2.CAP_PROP_FPS), 'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
            cap.release()
            assert probe == {'opened': True, 'fps': row['metadata']['src_fps'], 'frames': row['metadata']['src_frame_count'], 'width': 320, 'height': 240}
            t0 = time.perf_counter()
            for offset in range(0, len(pending), 4):
                batch = pending[offset:offset + 4]
                context['affected_keys'] = [make_clip_key(v, s) for v, s in batch]
                values = extract_batch(model, batch, index)
                for pair, value in zip(batch, values):
                    buffer_pairs.append(pair); buffer_features.append(value.clone())
                    if len(buffer_pairs) == 512:
                        flush()
                new_count += len(batch); context['observed_count'] += len(batch)
            after = raw.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            path = attempt / f'extraction_video_{number:03d}.json'
            write_json(path, {**row, 'realized_clip_keys': row['clip_keys'], 'raw_bytes': before.st_size,
                'raw_mtime_ns': before.st_mtime_ns, 'raw_probe': probe, 'wall_seconds': time.perf_counter() - t0,
                'retained_clips': len(row['clip_keys']) - len(pending), 'newly_extracted_clips': len(pending), 'attempt_id': 'attempt_002'})
            video_paths.append(str(path))
            print(json.dumps({'completed_videos': number + 1, 'persisted_or_new_clips': len(keys) + new_count, 'video_id': vid}), flush=True)
        flush()
        assert new_count == 57826
        for path, expected in prior.items():
            assert sha256_file(path) == expected, 'Original attempt changed'
        for r in read_json(attempt / 'recovery_plan.json')['retained_shards']:
            assert sha256_file(store / r['path']) == r['sha256'], 'Retained shard changed'
        identity = feature_store_identity(store, 'test')
        write_json(store / 'manifest.json', {'protocol_sha256': PROTOCOL_SHA, 'candidate_manifest_sha256': sha256_file(candidate_path),
            'candidate_manifest_path': str(candidate_path), 'identity': identity, 'shards': shards, 'clips': 69090, 'videos': 290,
            'validation_replay_sha256': sha256_file(attempt / 'validation_replay.json'), 'execution_sha256': sha256_file(attempt / 'execution.json'),
            'recovery_plan_sha256': sha256_file(attempt / 'recovery_plan.json')})
        all_keys, all_scores = [], []
        for path in sorted(store.glob('global_test_shard_*.pt')):
            shard = torch.load(path, map_location='cpu', weights_only=False)
            all_keys.extend(shard['clip_keys']); all_scores.append(teacher_scores(shard['global'], prototypes))
        scores = torch.cat(all_scores)
        assert all_keys == [k for r in candidates for k in r['clip_keys']] and torch.isfinite(scores).all()
        teacher_path = Path(outputs['teacher_root']) / 'teacher_scores_test.pt'
        save_tensor(teacher_path, {'clip_keys': all_keys, 'scores': scores, 'meta': {'prototype_sha256': sha256_file(PROTOTYPES),
            'feature_store_identity': identity['identity_sha256'], 'protocol_sha256': PROTOCOL_SHA}})
        write_json(attempt / 'extraction_complete.json', {'status': 'complete', 'clips': len(all_keys), 'videos': len(video_paths),
            'feature_store_identity': identity, 'teacher_sha256': sha256_file(teacher_path), 'candidate_manifest_path': str(candidate_path),
            'video_record_paths': video_paths, 'retained_clips': len(keys), 'new_clips': new_count, 'wall_seconds': time.perf_counter() - started,
            'max_allocated_bytes': torch.cuda.max_memory_allocated(), 'max_reserved_bytes': torch.cuda.max_memory_reserved(),
            'resolved_interruption': str(attempt / 'recovery_plan.json'), 'failures': []})
        print('PASS recovered extraction complete: 69090 clips / 290 videos', flush=True)
    except BaseException as exc:
        write_json(attempt / 'failure.json', {**context, 'code': 'recovery_failure', 'message': repr(exc)})
        raise


if __name__ == '__main__':
    main()
