# Stage-2 I/O Contract (DMAN v2.1)

## Immutable inputs (Stage-1 feature store)
Root:
- data/ucf_crime/artifacts/dman_v21/features/

Global shards:
- global_train_shard_*.pt
- global_val_shard_*.pt
- global_test_shard_*.pt

Each global shard is a torch-saved dict with:
- meta: dict
- clip_keys: List[str], length N
- video_ids: List[str], length N
- clip_starts: torch.int32 [N]
- labels: torch.int64 [N]   (0 Normal, 1 Anomaly)
- global: torch.float16 [N, 768]

Clip key format:
- "{rel_path}__s{clip_start:06d}"

Patch memory shards (Normal-train only, optional for Stage-2):
- patch_normal_train_mem_shard_*.pt
Shard schema:
- meta: dict
- clip_keys: List[str]
- video_ids: List[str]
- clip_starts: torch.int32 [N]
- patch: torch.float16 [N, 196, 768]

## Splits
- data/ucf_crime/splits/dman_v21/train.txt
- data/ucf_crime/splits/dman_v21/val.txt
- data/ucf_crime/splits/dman_v21/test.txt
Each is plain text: one rel video path per line.

## Labels index
- data/ucf_crime/artifacts/dman_v21/clip_index/videos.jsonl

Each line is a JSON object with at least:
- video_id (rel path)
- label_bin: 0/1
- group: class string (13 anomaly groups + Normal)

## Stage-2 requirements
- Stage-2 MUST NOT modify any Stage-1 artifact.
- Stage-2 MUST validate schema/dtypes/shapes before training.
- Stage-2 MUST be deterministic given a seed.
