"""Bounded Phase-4 GPU extraction benchmark; never runs a MIL checkpoint."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

# Use only the locally cached backbone; do not update/download model state.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import cv2
import torch
from stage1b_finetune.src.clip_dataset import decode_clip_frames, transform_clip
from stage1b_finetune.src.lora_backbone import LoRAVideoMAE
from stage2_class.src.utils.provenance import sha256_file

ROOT = Path('reproducibility/temporal_protocol')


def main():
    output = ROOT / 'benchmark_results.json'
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: host GPU access is required; no CPU substitution')
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    sample_path = ROOT/'benchmark_sample.json'
    plan = json.loads(sample_path.read_text())
    props = torch.cuda.get_device_properties(0)
    evidence = {'kind':'extraction_preflight_not_temporal_evaluation', 'sample_sha256':sha256_file(sample_path),
        'environment':{'python':platform.python_version(),'torch':torch.__version__, 'cuda_runtime':torch.version.cuda,
            'cudnn':torch.backends.cudnn.version(),'gpu':props.name,'dedicated_vram_bytes':props.total_memory,
            'shared_ram_counted_as_vram':False,'torch_cpu_threads':torch.get_num_threads(),
            'disk_free_bytes':shutil.disk_usage('.').free,
            'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)},
        'settings':{'batch_size':plan['batch_size'],'autocast':'float16','disk_dtype':'float16','warmup_batches':plan['warmup_batches'],
            'decode':'per-clip OpenCV seek, exact existing decode_clip_frames and transform_clip',
            'memory_policy':'only one batch of transformed clips retained; no full-video tensor'},
        'rows':[], 'failures':[]}
    t0=time.perf_counter()
    model=LoRAVideoMAE.load_from_checkpoint('artifacts/stage1b/checkpoints/lora_best.pt',device='cuda')
    torch.cuda.synchronize()
    evidence['model_load_seconds']=time.perf_counter()-t0
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
        warm=torch.zeros((plan['batch_size'],3,16,224,224),device='cuda')
        for _ in range(plan['warmup_batches']):
            model.extract_features(warm)
        torch.cuda.synchronize()
        del warm
        for repeat in range(plan['repeats']):
            for sample in plan['samples']:
                torch.cuda.reset_peak_memory_stats()
                row={'repeat':repeat,'video_id':sample['video_id'],'clips':0,'decode_transform_seconds':0.0,'gpu_forward_seconds':0.0}
                raw=Path('data/ucf_crime/raw')/sample['video_id']
                cap=cv2.VideoCapture(str(raw)); row['width']=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); row['height']=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
                t0=time.perf_counter()
                for offset in range(0,len(sample['starts']),plan['batch_size']):
                    t1=time.perf_counter(); tensors=[]
                    for start in sample['starts'][offset:offset+plan['batch_size']]:
                        frames=decode_clip_frames(str(raw),start,16,sample['metadata']['keep_every_n'])
                        if frames is None:
                            raise RuntimeError(f'Decode failure {raw}: {start}')
                        tensors.append(transform_clip(frames))
                    batch=torch.stack(tensors).to('cuda')
                    torch.cuda.synchronize(); row['decode_transform_seconds']+=time.perf_counter()-t1
                    t1=time.perf_counter()
                    features=model.extract_features(batch).to(torch.float16).cpu()
                    torch.cuda.synchronize(); row['gpu_forward_seconds']+=time.perf_counter()-t1
                    assert features.shape==(len(tensors),768) and torch.isfinite(features).all()
                    row['clips']+=len(tensors)
                    del features,batch,tensors,frames
                row['wall_seconds']=time.perf_counter()-t0
                row['max_allocated_bytes']=torch.cuda.max_memory_allocated()
                row['max_reserved_bytes']=torch.cuda.max_memory_reserved()
                row['cuda_free_bytes'],row['cuda_total_bytes']=torch.cuda.mem_get_info()
                evidence['rows'].append(row)
                print(json.dumps(row),flush=True)
    evidence['environment']['nvidia_smi_after']=subprocess.check_output(['nvidia-smi'],text=True)
    evidence['total_measured_clips']=sum(r['clips'] for r in evidence['rows'])
    evidence['total_measured_seconds']=sum(r['wall_seconds'] for r in evidence['rows'])
    evidence['max_allocated_bytes']=max(r['max_allocated_bytes'] for r in evidence['rows'])
    evidence['max_reserved_bytes']=max(r['max_reserved_bytes'] for r in evidence['rows'])
    count=json.loads((ROOT/'coverage_audit.json').read_text())['summary']['dense_candidate']['clips']
    rates=[r['wall_seconds']/r['clips'] for r in evidence['rows']]
    evidence['projection']={'test_candidate_clips':count,
        'weighted_seconds':count*evidence['total_measured_seconds']/evidence['total_measured_clips'],
        'slowest_sample_seconds':count*max(rates), 'feature_payload_bytes_fp16':count*768*2,
        'scope':'extrapolation, not measured full extraction; excludes final writes, teacher scoring, MIL and rendering'}
    with output.open('x') as handle:
        json.dump(evidence,handle,indent=2,allow_nan=False); handle.write('\n')
    print(json.dumps(evidence['projection']),flush=True)


if __name__=='__main__':
    main()
