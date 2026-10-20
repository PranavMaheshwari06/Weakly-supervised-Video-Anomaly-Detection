# Weakly Supervised Video Anomaly Understanding with Parameter-Efficient Adaptation and Temporal Grounding for Emergency Response

Code, trained models and reproducibility records for the master's thesis of Pranav Maheshwari (2026).

The thesis studies weakly supervised video anomaly detection on UCF-Crime, using video-level labels only. A LoRA-adapted VideoMAE encoder and a normal-prototype distance signal feed an attention multiple instance learning (MIL) model. The model has two heads: a binary anomaly head, and a three-channel response head that predicts which services (Police, Medical, Fire) an event would involve. The attention weights are evaluated separately as temporal localization. Saved model outputs are rendered as an offline CAD-style (computer-aided dispatch) alert.

```
raw video ─► Stage 0   fixed splits and clip index (16 frames at 15 fps, stride 8)
          ─► Stage 1   frozen VideoMAE features (pre-LoRA teacher, historical 14-class model)
          ─► Stage 1b  LoRA adaptation ─► adapted feature store ─► normal prototypes ─► teacher distance score
          ─► Stage 2   attention MIL: binary anomaly head + Police/Medical/Fire response head
          ─► Stage 3   binary-gated response routing ─► offline CAD-style alert
```

The stages are trained one after another. Gradients from Stage 2 never reach the video encoder.

## Contents

- [Results](#results)
- [Pipeline and thesis map](#pipeline-and-thesis-map)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data and trained artifacts](#data-and-trained-artifacts)
- [Verifying the reported results](#verifying-the-reported-results)
- [Running the model on a UCF-Crime video](#running-the-model-on-a-ucf-crime-video)
- [Re-running the pipeline](#re-running-the-pipeline)
- [Notes on the recorded evidence](#notes-on-the-recorded-evidence)

## Results

UCF-Crime test split (290 videos), three Stage-2 training seeds (1337, 2027, 3407), mean ± sample standard deviation. *Fixed bags* are immutable 16-clip bags per video. *Dense* uses every indexed clip of every test video (69,090 clips).

| Metric | Fixed bags | Dense |
|---|---|---|
| Binary AUROC (video level) | 0.940 ± 0.002 | 0.942 ± 0.001 |
| Binary F1 (anomaly class) | 0.875 ± 0.012 | 0.874 ± 0.010 |
| Response Macro-F1 (raw, threshold 0.5) | 0.478 ± 0.009 | |
| Response Hamming loss (raw) | 0.592 ± 0.033 | |
| Response exact match (raw) | 0.099 ± 0.020 | |
| Temporal AP (135 annotated anomalous videos) | | 0.327 ± 0.014 |
| Temporal IoU (fixed interval rule) | | 0.186 ± 0.017 |
| Source frames covered by indexed clips | | 99.843% |

The binary values are video-level and are not comparable with the frame-level AUC usually reported for UCF-Crime.

The main findings:
- Video-level anomaly detection is consistent across seeds and evaluation protocols.
- The raw response head has recall near one but low precision, and it predicts Police for every video.
- Binary gating improves the error profile. Even so, the learned gated Macro-F1 exceeds a binary-gated all-services reference by only 0.015.
- A validation-only study that adds normal bags to the response loss removes the universal Police activation, but does not improve service selection.
- Attention gives limited temporal alignment, not precise localization.

| Result | Evidence file |
|---|---|
| Fixed-bag binary and response metrics | [`reproducibility/evidence/dispatch_repro_v1/summary.json`](reproducibility/evidence/dispatch_repro_v1/summary.json) |
| Historical 14-class models | [`reproducibility/evidence/historical_14_class_v1/summary.json`](reproducibility/evidence/historical_14_class_v1/summary.json) |
| Normal-bag response-loss study (validation) | [`reproducibility/evidence/normal_bag_response_study/summary.json`](reproducibility/evidence/normal_bag_response_study/summary.json) |
| Teacher ablation (validation) | `artifacts/stage2/evidence/teacher_ablation_v1/summary.json` (evidence bundle) |
| Dense routing, temporal grounding, coverage | `artifacts/stage2/evidence/temporal_system_v1/summary.json` (evidence bundle) |
| Per-run validation and test evaluations | `artifacts/stage2/eval/<run_tag>/` (evidence bundle) |
| Thesis figures | `artifacts/thesis_figures/phase6/` (evidence bundle) |

[`reproducibility/PROVENANCE.md`](reproducibility/PROVENANCE.md) lists the SHA-256 hashes of every input and output behind these numbers, together with the recorded command that produced each of them.

## Pipeline and thesis map

| Stage | What it does | Code | Thesis section |
|---|---|---|---|
| 0 | Fixed train/val/test split (1,449 / 161 / 290 videos) and deterministic clip index | `stage0_data/` | Methods: *Dataset Partition and Deterministic Clip Index* |
| 1 | Frozen VideoMAE-Base clip features, normal selection, prototypes and clip scores. Used for the pre-LoRA teacher and the historical 14-class model | `stage1_features/`, `stage2_class/src/{preprocess,memory,teacher}/` | Methods: *VideoMAE Representation and Low-Rank Adaptation*. Results: *Historical 14-Class Development Outputs* |
| 1b | LoRA adaptation of VideoMAE, adapted feature store for all 1,900 videos, 100 normal prototypes, teacher distance score per clip | `stage1b_finetune/` | Methods: *VideoMAE Representation and Low-Rank Adaptation*; *Normal Prototype Memory and Distance Signal* |
| 2 | Attention MIL with shared attention, a binary head and a three-channel response head. The teacher score is appended to each clip embedding and guides training-bag construction | `stage2_class/src/{data,models,train}/`, `stage2_class/configs/hierA_dispatch_v1.yaml` | Methods: *Stage-2 Attention MIL Model*; *Training Objectives and Model Selection*. Results: *Official Fixed-Bag Binary Anomaly Results*; *Raw Three-Channel Response Results*; *Anomaly-Conditional and Binary-Gated Routing*; *Matched Trivial Baselines* |
| Studies | Teacher ablation and normal-bag response-loss study, both validation only | `stage2_class/configs/teacher_ablation_*.yaml`, `stage2_class/configs/abl_normal_response*_v1.yaml` | Methods: *Validation-Only Teacher-Mechanism Study*; *Post-Hoc Normal-Bag Response-Loss Study*. Results: *Validation-Only Teacher Ablation*; *Normal-Bag Response-Loss Study (Validation Only)* |
| Evaluation | Fixed-bag and dense indexed evaluation | `stage2_class/src/eval/`, `utils/evaluate_temporal_system.py` | Methods: *Fixed-Bag and Dense Indexed Evaluation* |
| Temporal | Frame-level AP and fixed-rule IoU of the attention weights under a frozen protocol | `stage2_class/src/eval/temporal_*.py`, `utils/*temporal*.py`, `reproducibility/temporal_protocol/` | Methods: *Temporal Grounding Protocol*. Results: *Dense Indexed-System and Temporal-Grounding Results* |
| 3 | Binary-gated routing, CAD-style alert record and annotated video | `stage3_dispatch/`, `inference/inference_any_video.py` | Methods: *Offline CAD-Style Output*. Results: *Qualitative Offline CAD-Style Example* |

Configurations behind the reported runs:

| Result | Config (`stage2_class/configs/`) | Run tags |
|---|---|---|
| Main three-seed results | `hierA_dispatch_v1.yaml` | `dispatch_repro_s{1337,2027,3407}` |
| Teacher ablation (validation only) | `teacher_ablation_{full,neither,no_input,uniform}.yaml` | `ablation_*_s{1337,2027,3407}` |
| Normal-bag response-loss study (validation only) | `abl_normal_response_v1.yaml`, `abl_normal_response_reweighted_v1.yaml` | `abl_normal_response[_reweighted]_v1_s*` |
| Historical 14-class models | `hierA_dual_head.yaml`, `hierA_lora_features.yaml` | `hier_cascaded_v3`, `hier_lora_e2e` |
| Dense temporal evaluation | frozen protocol in `reproducibility/temporal_protocol/` | `temporal_system_v1/attempt_002` |

`base.yaml` is the default configuration read by the Stage 1 and Stage 2 preprocessing scripts.

## Repository layout

| Path | Content |
|---|---|
| `stage0_data/` | Video indexing, decoding, split construction and clip index. [`stage0_data/README.md`](stage0_data/README.md) documents the data contract |
| `stage1_features/` | Frozen VideoMAE backbone and feature extraction |
| `stage1b_finetune/` | LoRA adaptation, adapted-feature extraction, LoRA prototypes and teacher scores |
| `stage2_class/` | MIL datasets, model, training, evaluation, configs and unit tests |
| `stage3_dispatch/` | Response routing and CAD alert video rendering |
| `inference/` | Inference and CAD alert for one indexed UCF-Crime video |
| `utils/` | Dense temporal extraction and evaluation, evidence builders and verifiers, figure generation |
| `configs/` | Shared Stage 0/1 configuration |
| `data/ucf_crime/splits/` | Official UCF-Crime split lists and the fixed split used in the thesis |
| `reproducibility/` | Provenance record, environment record, evidence mirrors, frozen temporal protocol, recorded source for the teacher-ablation check |
| `run_dispatch_pipeline.sh` | Stage 2 workflow for one seed: evaluation bags, training, validation evaluation |

## Installation

Tested on Ubuntu 24.04 (WSL2) with Python 3.12.3 and one NVIDIA RTX 4050 Laptop GPU (6 GB). Verification and single-video inference also run on CPU.

```bash
git clone https://github.com/PranavMaheshwari06/Weakly-supervised-Video-Anomaly-Detection.git
cd Weakly-supervised-Video-Anomaly-Detection
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

All commands below are run from the repository root with this environment active.

## Data and trained artifacts

No large file is stored in Git.

1. **Evidence bundle** (`thesis-evidence-core-v1.tar.gz`, 0.35 GB, about 0.5 GB unpacked). It is attached to the [v1.0 release](https://github.com/PranavMaheshwari06/Weakly-supervised-Video-Anomaly-Detection/releases/tag/v1.0) together with `SHA256SUMS`. Unpack it at the repository root; it restores the original paths under `artifacts/`, `data/ucf_crime/` and `legacy/`, all of which Git ignores.

   ```bash
   gh release download v1.0      # or download both files from the release page
   sha256sum -c SHA256SUMS
   tar -xzf thesis-evidence-core-v1.tar.gz
   ```

   The bundle contains:
   - the clip index and the LoRA-adapted feature store for all 1,900 videos;
   - the LoRA checkpoint, the normal prototypes and the teacher scores;
   - the temporal test annotations;
   - all Stage-2 run directories, including the three `dispatch_repro_*` checkpoints;
   - the fixed evaluation bags and the per-run evaluations;
   - the dense temporal outputs, the evidence summaries and the thesis figures.

2. **UCF-Crime videos** (about 97 GB). These are not redistributed. Download them from the dataset authors (Sultani, Chen and Shah, CVPR 2018; https://www.crcv.ucf.edu/projects/real-world/) and place them under `data/ucf_crime/raw/`, one folder per category (`Abuse/`, …, `Vandalism/`, `Training_Normal_Videos_Anomaly/`, `Testing_Normal_Videos_Anomaly/`), as in `data/ucf_crime/splits/dman_v21/*.txt`. The videos are needed only for single-video inference and for re-running Stages 0 to 1b.

3. **Frozen VideoMAE feature store** (about 15 GB, `data/ucf_crime/artifacts/dman_v21/features/`). It is needed only to re-run the pre-LoRA stage and the historical 14-class model, and to recompute its shard identity. Available on request.

## Verifying the reported results

These checks need the evidence bundle. They do not need a GPU or the raw videos. The verifiers recompute file hashes and metrics from the stored per-video outputs and compare them with the recorded evidence.

```bash
# unit tests (these do not need the bundle)
python -m unittest stage2_class.tests.test_reproducibility stage2_class.tests.test_teacher_ablation \
    stage2_class.tests.test_temporal_geometry stage2_class.tests.test_temporal_system

# main fixed-bag results, data split and feature-store records
python -B utils/verify_thesis_evidence.py --skip-feature-stores   # drop the flag if the frozen feature store is present

# historical 14-class models
python -B utils/verify_historical_14_class_evidence.py

# dense evaluation, temporal grounding and CAD records of all three checkpoints
python -B -m utils.verify_temporal_preflight
python -B -m utils.verify_temporal_system --attempt-id attempt_002 --skip-raw-video-check

# complete evidence package and thesis figures
python -B -m utils.verify_canonical_evidence_package --require-figures
```

Each verifier ends with `PASS` or `"status": "pass"`. `--skip-raw-video-check` skips only the comparison of raw-video file sizes and modification times with the extraction records, which can match only on the original machine.

**Teacher ablation.** `utils/verify_teacher_ablation_evidence.py` pins the hash of the trainer as it was when the ablation ran. The trainer later gained the two normal-bag study options, which are off by default (see the notes below). Run the check against the recorded trainer, then restore the current one:

```bash
cp reproducibility/recorded_sources/m27_train_dispatch_mil.py stage2_class/src/train/
python -B utils/verify_teacher_ablation_evidence.py   # PASS: 12 runs, 84 artifacts, 48 recomputed metrics
git checkout stage2_class/src/train/m27_train_dispatch_mil.py
```

## Running the model on a UCF-Crime video

`inference/inference_any_video.py` runs a trained Stage-2 checkpoint on one UCF-Crime video. It prints the CAD-style alert and writes an annotated copy of the video. It needs the evidence bundle and the original video file at its place under `data/ucf_crime/raw/`. It runs on CPU in a few seconds per video.

The example below reproduces the CAD figure in the thesis (seed 1337 checkpoint, test video `Abuse028`). The bundle already holds the thesis output for this video, so it is copied first for comparison:

```bash
# needs data/ucf_crime/raw/Abuse/Abuse028_x264.mp4
cp artifacts/inference/dispatch_repro_s1337/Abuse028_x264_metadata.json /tmp/thesis_Abuse028_metadata.json

python inference/inference_any_video.py --video_id "Abuse/Abuse028_x264" \
    --run_tag dispatch_repro_s1337 --mode dispatch

diff /tmp/thesis_Abuse028_metadata.json artifacts/inference/dispatch_repro_s1337/Abuse028_x264_metadata.json \
    && echo "identical to the thesis output"
```

Expected alert: anomaly detected, services POLICE and MEDICAL, priority P1-MEDICAL.

| Output | Path |
|---|---|
| Model output per clip (scores, attention, intervals) | `artifacts/inference/<run_tag>/<video>_metadata.json` |
| CAD-style alert record | `artifacts/inference/<run_tag>/<video>_alert.json` |
| Annotated video | `artifacts/visualizations/<run_tag>/<video>_dispatch.mp4` |

- **Other videos.** Any other UCF-Crime video works the same way. Pass `--video_id "<Category>/<file name without .mp4>"`, for example `"Explosion/Explosion017_x264"`.
- **Other checkpoints.** `--run_tag dispatch_repro_s2027` or `dispatch_repro_s3407` selects one of the other two official checkpoints.
- **Re-running Abuse028.** This rewrites the bundled alert record for that video with a new timestamp; the metadata is unchanged.

**Scope.** The script uses the precomputed adapted features of the indexed video and checks the source file's frame count and frame rate against the clip index. This is the dense indexed system evaluated in the thesis. Videos outside UCF-Crime, or re-encoded copies, are not supported, because the thesis evaluates no raw-video-to-feature path for them. The CAD output is an offline research demonstration with no calibration, safety or operational validation.

## Re-running the pipeline

Each step writes to a run tag or output directory. Existing result directories are protected, so use a new run tag. Stages 0 to 1b need the raw videos and a GPU.

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
```

Stage 1 and Stage 1b download `MCG-NJU/videomae-base` from Hugging Face on first use.

The LoRA feature extraction used process-dependent sampling, so a re-run produces a new feature store rather than the recorded one. Stage 2 results are therefore reproducible given the provided feature store.

## Notes on the recorded evidence

- **Test split.** The UCF-Crime test split was evaluated after model selection. The teacher ablation and the normal-bag study are validation only, and their configurations refuse test evaluation.
- **Commit identifiers and folder names.** Run manifests and evidence records were written during development. Commit identifiers in them refer to the development repository; this repository is a clean snapshot of its final state.
  - Some records name their original folder locations (for example `RP_docs/evidence/` or `final_thesis_execution/phase_04_temporal_design_compute_preflight/`) and use internal phase numbers.
  - The verifiers map those locations to `reproducibility/`. The records themselves are unchanged so that their hashes stay valid.
- **Temporal extraction script.** `utils/extract_temporal_system.py` is byte-identical to the version recorded in the temporal evidence. To re-run it, set its `PHASE4` constant to `reproducibility/temporal_protocol`.
- **Trainer options and the recorded copy.**
  - `stage2_class/src/train/m27_train_dispatch_mil.py` gained the options `train.normal_response_loss` and `train.normal_response_weight` for the normal-bag study after the teacher ablation was recorded.
  - Both options are off by default and in every other configuration, so the training path of the official and ablation runs is unchanged.
  - `reproducibility/recorded_sources/m27_train_dispatch_mil.py` is the earlier version, with SHA-256 `166c69ab…74cc9b` as pinned by the teacher-ablation evidence.
- **Skipped inventory files.** `utils/verify_thesis_evidence.py` skips two inventoried files that are not part of this repository and prints the reason: `requirements.txt` was replaced by the pinned file, and a development report was not included.
