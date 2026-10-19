"""Validation-only feature replay to audit Transformers VideoMAE bias migration."""
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'

import torch
from safetensors.torch import load_file
from stage1b_finetune.src.lora_backbone import LoRAVideoMAE
from stage1b_finetune.src.clip_dataset import decode_clip_frames, transform_clip
from stage2_class.src.utils.provenance import sha256_file

ROOT=Path('reproducibility/temporal_protocol')
CACHE=Path.home()/'.cache/huggingface/hub/models--MCG-NJU--videomae-base/snapshots/dc740ceda42fce44faed2ea03c6d447db72f6af9'


def restore_pretrained_biases(model):
    """Map original q_bias/v_bias into new Linear biases; old key bias was zero."""
    weights=load_file(str(CACHE/'model.safetensors'))
    with torch.no_grad():
        for i, layer in enumerate(model.backbone.base_model.model.encoder.layer):
            attn=layer.attention.attention
            for old, new in [('q_bias',attn.query.base_layer.bias),('v_bias',attn.value.base_layer.bias)]:
                new.copy_(weights[f'videomae.encoder.layer.{i}.attention.attention.{old}'])
            attn.key.bias.zero_()


def main():
    out=ROOT/'backbone_replay.json'
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(4); torch.manual_seed(1337)
    assert torch.cuda.is_available()
    samples=json.loads((ROOT/'benchmark_sample.json').read_text())['samples']
    sparse={}
    sample_ids={s['video_id'] for s in samples}
    for path in sorted(Path('data/ucf_crime/artifacts/dman_v21/features_lora').glob('global_val_shard_*.pt')):
        d=torch.load(path,map_location='cpu',weights_only=False)
        for i,(vid,start) in enumerate(zip(d['video_ids'],d['clip_starts'].tolist())):
            if vid in sample_ids: sparse.setdefault(vid,[]).append((start,d['global'][i].float()))
    tensors=[]; targets=[]; rows=[]
    for sample in samples:
        clips=sorted(sparse[sample['video_id']]); chosen=sorted(set([0,len(clips)//2,len(clips)-1]))
        for i in chosen:
            start,expected=clips[i]
            frames=decode_clip_frames(str(Path('data/ucf_crime/raw')/sample['video_id']),start,16,sample['metadata']['keep_every_n'])
            assert frames is not None
            tensors.append(transform_clip(frames)); targets.append(expected)
            rows.append({'video_id':sample['video_id'],'start':start})
    model=LoRAVideoMAE.load_from_checkpoint('artifacts/stage1b/checkpoints/lora_best.pt',device='cuda')
    results={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for mode in ('current_loader','restored_pretrained_biases'):
            if mode=='restored_pretrained_biases': restore_pretrained_biases(model)
            values=[]
            for i in range(0,len(tensors),4):
                values.extend(model.extract_features(torch.stack(tensors[i:i+4]).cuda()).half().cpu().float())
            records=[]
            for row,actual,expected in zip(rows,values,targets):
                records.append({**row,'max_abs_error':float((actual-expected).abs().max()),
                    'mean_abs_error':float((actual-expected).abs().mean()),
                    'cosine_similarity':float(torch.nn.functional.cosine_similarity(actual[None],expected[None]))})
            results[mode]=records
    payload={'kind':'validation_feature_replay_no_MIL_or_temporal_results','backbone_cache_revision':CACHE.name,
        'backbone_hashes':{p.name:sha256_file(p) for p in CACHE.iterdir()},'results':results}
    with out.open('x') as f: json.dump(payload,f,indent=2); f.write('\n')
    print(json.dumps({mode:{'max_error':max(r['max_abs_error'] for r in vals),'min_cosine':min(r['cosine_similarity'] for r in vals)} for mode,vals in results.items()}))


if __name__=='__main__': main()
