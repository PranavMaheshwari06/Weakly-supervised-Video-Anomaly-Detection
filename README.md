# Weakly Supervised Video Anomaly Understanding with Parameter-Efficient Adaptation and Temporal Grounding for Emergency Response

Code and reproducibility records for the master's thesis of Pranav Maheshwari.

The pipeline detects anomalous videos in UCF-Crime under video-level (weak) supervision and predicts which emergency services (Police, Medical, Fire) a detected event would involve. It also evaluates how well the model's attention locates the annotated anomalous interval, and renders an offline CAD-style (computer-aided dispatch) alert.

```
raw video ─► Stage 0: decoding, fixed splits, clip index (16 frames @ 15 fps, stride 8)
          ─► Stage 1: frozen VideoMAE features ─► normal prototypes ─► teacher distance score
          ─► Stage 1b: LoRA adaptation of VideoMAE ─► adapted features ─► LoRA normal prototypes + teacher scores
          ─► Stage 2: attention MIL with a binary anomaly head and a three-channel response head
          ─► Stage 3: binary-gated response routing ─► CAD-style alert (offline)
```

The stages were optimised one after another; gradients from Stage 2 never reach the video encoder.

## Repository layout

| Path | Content |
|---|---|
| `stage0_data/` | Video indexing, decoding, split construction, clip index (`README.md` documents the data contract) |
| `stage1_features/` | Frozen VideoMAE backbone and feature extraction (used for the pre-LoRA teacher and the historical 14-class model) |
| `stage1b_finetune/` | LoRA adaptation, adapted-feature extraction, LoRA prototypes and teacher scores |
| `stage2_class/` | MIL datasets, model, training, fixed-bag evaluation, configs, tests |
| `stage3_dispatch/` | Response routing and CAD alert video rendering |
| `inference/` | Single-video inference for an indexed video |
| `utils/` | Dense temporal extraction/evaluation, evidence builders and verifiers, thesis figure generation |
| `configs/` | Shared Stage 0/1 configuration |
| `data/ucf_crime/splits/` | Official UCF-Crime split lists and the fixed train/val/test split used in the thesis |
| `reproducibility/` | `PROVENANCE.md` (hashes and commands for every reported result), `ENVIRONMENT.md`, small evidence mirrors, frozen temporal protocol |

### Configurations behind the reported results

| Result in the thesis | Config (`stage2_class/configs/`) | Run tags |
|---|---|---|
| Main three-seed results | `hierA_dispatch_v1.yaml` | `dispatch_repro_s{1337,2027,3407}` |
| Teacher ablation (validation only) | `teacher_ablation_{full,neither,no_input,uniform}.yaml` | `ablation_*_s{1337,2027,3407}` |
| Normal-bag response-loss study (validation only) | `abl_normal_response_v1.yaml`, `abl_normal_response_reweighted_v1.yaml` | `abl_normal_response[_reweighted]_v1_s*` |
| Historical 14-class models (motivation section) | `hierA_dual_head.yaml`, `hierA_lora_features.yaml` | `hier_cascaded_v3`, `hier_lora_e2e` |
| Dense temporal evaluation | frozen protocol in `reproducibility/temporal_protocol/` | `temporal_system_v1/attempt_002` |

`base.yaml` is the default configuration read by the Stage 1/2 preprocessing scripts.

## Installation

Tested on Ubuntu 24.04 (WSL2), Python 3.12.3, one NVIDIA RTX 4050 Laptop GPU (6 GB).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

## Data

Nothing large is stored in Git.

1. **UCF-Crime videos (~97 GB).** Not redistributed. Download from the dataset authors (Sultani, Chen and Shah, CVPR 2018; https://www.crcv.ucf.edu/projects/real-world/) and arrange as described in `stage0_data/README.md` under `data/ucf_crime/raw/` and `data/ucf_crime/annotations/`. The fixed splits are already in `data/ucf_crime/splits/`.
2. **Evidence bundle** `thesis-evidence-core-v1.tar.gz` (~0.35 GB, ~0.5 GB unpacked; provided separately). Clip index, adapted feature stores, LoRA checkpoint, prototypes, teacher scores, the temporal annotations, and the run, evaluation and evidence files of every reported result. Unpack it at the repository root (`tar -xzf thesis-evidence-core-v1.tar.gz`); this restores the original paths under `data/ucf_crime/` and `artifacts/`.
3. **Frozen VideoMAE feature store** (~15 GB, `data/ucf_crime/artifacts/dman_v21/features/`). Needed only to re-run the pre-LoRA stage and the historical 14-class model, and to recompute its shard identity. Available on request.

## Checking the reported results (no GPU, no raw video)

With the evidence bundle unpacked:

```bash
python -m unittest stage2_class.tests.test_reproducibility stage2_class.tests.test_teacher_ablation \
    stage2_class.tests.test_temporal_geometry stage2_class.tests.test_temporal_system
python -B utils/verify_thesis_evidence.py --skip-feature-stores   # drop the flag if the frozen feature store is present
python -B utils/verify_historical_14_class_evidence.py
python -B utils/verify_teacher_ablation_evidence.py
python -B -m utils.verify_temporal_preflight
python -B -m utils.verify_temporal_system --attempt-id attempt_002 --skip-raw-video-check
python -B -m utils.verify_canonical_evidence_package --require-figures
```

The verifiers recompute file hashes and metrics from the stored per-video outputs and compare them with the recorded evidence. `--skip-raw-video-check` skips only the comparison of raw-video file size and modification time with the extraction records, which can match only on the original machine. The unit tests do not need the bundle.

## Re-running the pipeline

Each step writes to a run tag or output directory. Existing result directories are protected; use a new run tag.

```bash
# Stage 0: splits and clip index
python -m stage0_data.src.split_builder
python -m stage0_data.src.clip_index

# Stage 1 (pre-LoRA teacher): frozen features, normal selection, prototypes, clip scores
python -m stage1_features.src.feature_dump --help
python -m stage2_class.src.preprocess.m21_normal_selection
python -m stage2_class.src.memory.m22_build_global_memory
python -m stage2_class.src.teacher.m23_score_clips

# Stage 1b: LoRA adaptation, adapted features (train/val/test), LoRA prototypes and teacher scores
python -m stage1b_finetune.src.train_lora --cfg stage1b_finetune/configs/lora_finetune.yaml
python -m stage1b_finetune.src.extract_lora_features --cfg stage1b_finetune/configs/lora_finetune.yaml \
    --lora_ckpt artifacts/stage1b/checkpoints/lora_best.pt --split train   # then val, test
python -m stage1b_finetune.src.retrain_teacher --cfg stage1b_finetune/configs/lora_finetune.yaml

# Stage 2: fixed evaluation bags, training and validation evaluation for one seed
bash run_dispatch_pipeline.sh <RUN_TAG> <SEED>
STAGE2_CFG=stage2_class/configs/hierA_dispatch_v1.yaml \
    python -m stage2_class.src.eval.m29_eval_dispatch --run_tag <RUN_TAG> --split test
python -m stage2_class.src.eval.m31_aggregate_dispatch_repro      # summary of the three official seeds

# Dense temporal evaluation (recorded commands; output locations are set by the frozen protocol)
python -m utils.extract_temporal_system
python -m utils.resume_temporal_extraction
python -m utils.evaluate_temporal_system --device cuda --attempt-id attempt_002

# CAD alert for one indexed video
python inference/inference_any_video.py --video_id "Abuse/Abuse028_x264" --run_tag <RUN_TAG> --mode dispatch
```

`reproducibility/PROVENANCE.md` gives the exact recorded command for every reported result together with the hashes of its inputs and outputs. The LoRA feature extraction used process-dependent sampling, so a re-run produces a new feature store rather than the recorded one; Stage 2 results are therefore reproducible conditional on the provided feature store.

## Notes on the recorded evidence

- Run manifests and evidence records were written during development. Commit identifiers in them refer to the development repository; this repository is a clean snapshot of its final state. Some frozen records also name their original folder locations (for example `RP_docs/evidence/` or `final_thesis_execution/phase_04_temporal_design_compute_preflight/`) and use the project's internal phase numbers. The verifiers map those locations to `reproducibility/`; the records themselves are left unchanged so that their hashes stay valid.
- `utils/extract_temporal_system.py` is kept byte-identical to the version recorded in the released temporal evidence. To re-run it, set its `PHASE4` constant to `reproducibility/temporal_protocol`.
- `stage2_class/src/train/m27_train_dispatch_mil.py` gained the two `train.normal_response_*` options for the normal-bag study after the teacher ablation had been recorded. Both options are off by default and in every other configuration, so the training path of the official and ablation runs is unchanged. However, `utils/verify_teacher_ablation_evidence.py` pins the file hash from before this addition and reports `implementation hash` for this file. With the earlier version of the trainer it passes in full (12 runs, 84 artifacts, 48 recomputed metrics).
- `utils/verify_thesis_evidence.py` skips two inventoried files that are not part of this repository and prints the reason (`requirements.txt` was replaced by the pinned file; a development report was not included).
- The UCF-Crime test split was evaluated after model selection. The teacher ablation and the normal-bag study are validation-only; their configurations refuse test evaluation.
