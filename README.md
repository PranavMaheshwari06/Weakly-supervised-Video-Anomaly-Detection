# Weakly Supervised Video Anomaly Understanding with Parameter-Efficient Adaptation and Temporal Grounding for Emergency Response

Code, trained models and reproducibility records for my master's thesis (Pranav Maheshwari, 2026).

The thesis looks at weakly supervised video anomaly detection on UCF-Crime, using only video-level labels. A LoRA-adapted VideoMAE encoder and a normal-prototype distance signal feed an attention multiple instance learning (MIL) model with two heads: a binary anomaly head, and a three-channel response head that predicts which services (Police, Medical, Fire) an event would involve. The attention weights are evaluated separately as temporal localization, and saved model outputs are rendered as an offline CAD-style (computer-aided dispatch) alert.

The stages are trained one after another. Stage 0 builds the fixed splits and the clip index (16 frames at 15 fps, stride 8). Stage 1 extracts frozen VideoMAE features for the pre-LoRA teacher and the historical 14-class model. Stage 1b adapts VideoMAE with LoRA and builds the normal prototypes and teacher distance score. Stage 2 trains the attention MIL model, and Stage 3 routes the response by the binary decision and writes the alert. Gradients from Stage 2 never reach the video encoder.

Not everything can be rerun from this repo alone. Checking the reported results needs the evidence bundle (see below) but no GPU and no videos. Running the model on a video also needs that video. Retraining Stages 0 to 1b needs the raw videos and a GPU, and the 15 GB frozen feature store is only available on request. The LoRA feature extraction is not deterministic, so a retrain gives a new feature store rather than the recorded one.

## Quick check

Checking the reported results needs no GPU, no training and no UCF-Crime videos:

```bash
git clone https://github.com/PranavMaheshwari06/Weakly-supervised-Video-Anomaly-Detection.git
cd Weakly-supervised-Video-Anomaly-Detection
gh release download v1.1                 # or download both files from the v1.1 release page into this folder
sha256sum -c SHA256SUMS && tar -xzf thesis-evidence-core-v1.tar.gz
python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
export PYTHONPATH=.
python -B utils/verify_thesis_evidence.py --skip-feature-stores    # ends with PASS
```

To run the trained model on one video, add that UCF-Crime video under `data/ucf_crime/raw/` and follow [Running the model on a UCF-Crime video](#running-the-model-on-a-ucf-crime-video). The checkpoints are in the bundle, so nothing has to be trained.

## Results

All numbers are on the UCF-Crime test split (290 videos), over three Stage-2 training seeds (1337, 2027, 3407), as mean ± sample standard deviation. Fixed bags are immutable 16-clip bags per video. Dense means every indexed clip of every test video (69,090 clips).

- Binary AUROC (video level): 0.940 ± 0.002 on fixed bags, 0.942 ± 0.001 dense.
- Binary F1 (anomaly class): 0.875 ± 0.012 on fixed bags, 0.874 ± 0.010 dense.
- Response head on fixed bags, raw at threshold 0.5: Macro-F1 0.478 ± 0.009, Hamming loss 0.592 ± 0.033, exact match 0.099 ± 0.020.
- Temporal grounding, dense, on the 135 annotated anomalous videos: AP 0.327 ± 0.014 and IoU with the fixed interval rule 0.186 ± 0.017.
- Source frames covered by the indexed clips: 99.843%.

The binary values are video-level, so they can't be compared with the frame-level AUC usually reported for UCF-Crime.

I also computed paired video-level bootstrap intervals (95% percentile, 10,000 resamples, mean over the three seeds per resample). They only cover video sampling, not training variance. Binary AUROC: 0.910 to 0.967. Binary-gated learned Macro-F1 minus the binary-gated all-services reference: 0.008 to 0.023. Temporal AP minus the whole-video reference: 0.110 to 0.164.

What this shows, in short:
- Video-level anomaly detection is consistent across seeds and evaluation protocols.
- The raw response head has recall near one but low precision, and it predicts Police for every video.
- Binary gating improves the error profile, but the learned gated Macro-F1 is only 0.015 above a binary-gated all-services reference.
- A validation-only study that adds normal bags to the response loss removes the universal Police activation, but does not improve service selection.
- Attention gives limited temporal alignment, not precise localization.

Where the numbers come from:
- Fixed-bag binary and response metrics: [`reproducibility/evidence/dispatch_repro_v1/summary.json`](reproducibility/evidence/dispatch_repro_v1/summary.json)
- Historical 14-class models: [`reproducibility/evidence/historical_14_class_v1/summary.json`](reproducibility/evidence/historical_14_class_v1/summary.json)
- Normal-bag response-loss study (validation): [`reproducibility/evidence/normal_bag_response_study/summary.json`](reproducibility/evidence/normal_bag_response_study/summary.json)
- Prototype-distance study (validation, called `teacher_ablation` in the code): `artifacts/stage2/evidence/teacher_ablation_v1/summary.json`, in the evidence bundle
- Dense routing, temporal grounding and coverage: `artifacts/stage2/evidence/temporal_system_v1/summary.json`, in the evidence bundle
- Bootstrap intervals: [`reproducibility/evidence/bootstrap_v1/bootstrap_cis.json`](reproducibility/evidence/bootstrap_v1/bootstrap_cis.json)
- Per-run validation and test evaluations: `artifacts/stage2/eval/<run_tag>/`, in the evidence bundle
- Thesis figures: `artifacts/thesis_figures/phase6/` in the bundle, and the final ones in [`thesis_final/images/`](thesis_final/images/)

[`reproducibility/PROVENANCE.md`](reproducibility/PROVENANCE.md) lists the SHA-256 hashes of every input and output behind these numbers, with the recorded command that produced each.

## How the code maps to the thesis

- Stage 0 (`stage0_data/`): fixed train/val/test split (1,449 / 161 / 290 videos) and the deterministic clip index. Thesis: Methods, *Dataset Partition and Deterministic Clip Index*.
- Stage 1 (`stage1_features/`, `stage2_class/src/{preprocess,memory,teacher}/`): frozen VideoMAE-Base clip features, normal selection, prototypes and clip scores, used for the pre-LoRA teacher and the historical 14-class model. Thesis: Methods, *VideoMAE Representation and Low-Rank Adaptation*; Results, *Historical 14-Class Development Outputs*.
- Stage 1b (`stage1b_finetune/`): LoRA adaptation, the adapted feature store for all 1,900 videos, 100 normal prototypes and the teacher distance score per clip. Thesis: Methods, *VideoMAE Representation and Low-Rank Adaptation* and *Normal Prototype Memory and Distance Signal*.
- Stage 2 (`stage2_class/src/{data,models,train}/`, `stage2_class/configs/hierA_dispatch_v1.yaml`): attention MIL with shared attention, a binary head and a three-channel response head. The teacher score is appended to each clip embedding and guides how training bags are built. Thesis: Methods, *Stage-2 Attention MIL Model* and *Training Objectives and Model Selection*; Results, *Official Fixed-Bag Binary Anomaly Results*, *Raw Three-Channel Response Results* and *Binary-Gated Routing and Matched Baselines*.
- The two validation-only studies (`stage2_class/configs/teacher_ablation_*.yaml`, `stage2_class/configs/abl_normal_response*_v1.yaml`): the prototype-distance study and the normal-bag response-loss study. Thesis: Methods and Results, *Validation-Only Prototype-Distance Study*, *Post-Hoc Normal-Bag Response-Loss Study* and *Normal-Bag Response-Loss Study (Validation Only)*.
- Evaluation (`stage2_class/src/eval/`, `utils/evaluate_temporal_system.py`): fixed-bag and dense indexed evaluation. Thesis: Methods, *Fixed-Bag and Dense Indexed Evaluation*.
- Temporal grounding (`stage2_class/src/eval/temporal_*.py`, `utils/*temporal*.py`, `reproducibility/temporal_protocol/`): frame-level AP and fixed-rule IoU of the attention weights under a frozen protocol. Thesis: Methods, *Temporal Grounding Protocol*; Results, *Dense Indexed-System and Temporal-Grounding Results* and *Qualitative Success and Failure Cases*.
- Stage 3 (`stage3_dispatch/`, `inference/inference_any_video.py`): binary-gated routing, the CAD-style alert record and the annotated video. Thesis: Methods, *Offline CAD-Style Output*; Results, *Offline CAD-Style Example*.
- Statistics and figures (`utils/bootstrap_test_cis.py`, `utils/generate_revision_figures.py`, `utils/generate_phase6_figures.py`): the paired bootstrap intervals and the print-size figures, both computed from the canonical evidence package.

Configs behind the reported runs (all in `stage2_class/configs/`):
- Main three-seed results: `hierA_dispatch_v1.yaml`, run tags `dispatch_repro_s{1337,2027,3407}`.
- Prototype-distance study (validation only): `teacher_ablation_{full,neither,no_input,uniform}.yaml`, tags `ablation_*_s{1337,2027,3407}`.
- Normal-bag response-loss study (validation only): `abl_normal_response_v1.yaml` and `abl_normal_response_reweighted_v1.yaml`, tags `abl_normal_response[_reweighted]_v1_s*`.
- Historical 14-class models: `hierA_dual_head.yaml` and `hierA_lora_features.yaml`, tags `hier_cascaded_v3` and `hier_lora_e2e`.
- Dense temporal evaluation: the frozen protocol in `reproducibility/temporal_protocol/`, run `temporal_system_v1/attempt_002`.

`base.yaml` is the default config read by the Stage 1 and Stage 2 preprocessing scripts.

## Repository layout

- `stage0_data/`: video indexing, decoding, split construction and clip index. [`stage0_data/README.md`](stage0_data/README.md) describes the data layout it expects.
- `stage1_features/`: frozen VideoMAE backbone and feature extraction.
- `stage1b_finetune/`: LoRA adaptation, adapted-feature extraction, LoRA prototypes and teacher scores.
- `stage2_class/`: MIL datasets, model, training, evaluation, configs and unit tests (in `stage2_class/tests/`).
- `stage3_dispatch/`: response routing and CAD alert video rendering.
- `inference/`: inference and CAD alert for one indexed UCF-Crime video.
- `utils/`: dense temporal extraction and evaluation, evidence builders and verifiers, bootstrap intervals, figure generation.
- `configs/`: the shared Stage 0/1 config.
- `data/ucf_crime/splits/`: the official UCF-Crime split lists and the fixed split used in the thesis.
- `reproducibility/`: provenance and environment records, evidence copies (including the bootstrap intervals), the frozen temporal protocol, and the recorded source for the prototype-distance check.
- `thesis_final/`: LaTeX sources of the thesis (`thesis.tex`, chapters, front matter, `bibliography.bib`) and the figures it includes.
- `run_dispatch_pipeline.sh`: the Stage 2 workflow for one seed (evaluation bags, training, validation evaluation).

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

No large file is stored in Git. Three things are needed, depending on what you want to do.

The evidence bundle is `thesis-evidence-core-v1.tar.gz` (0.35 GB, about 0.5 GB unpacked). It is attached to the [`v1.1` release of this repository](https://github.com/PranavMaheshwari06/Weakly-supervised-Video-Anomaly-Detection/releases/tag/v1.1) together with `SHA256SUMS`. The release tag is `v1.1` while the archive name still says `v1`; they are the same bundle. The repository is private, so the download needs a GitHub account with access (or `gh release download v1.1`). Put both files at the repository root and unpack the archive there. It restores the original paths under `artifacts/`, `data/ucf_crime/` and `legacy/`, all of which Git ignores.

```bash
sha256sum -c SHA256SUMS
tar -xzf thesis-evidence-core-v1.tar.gz
```

The bundle was built before the final revision, so it doesn't have the bootstrap output or the print-size revision figures. The bootstrap intervals are mirrored under `reproducibility/evidence/bootstrap_v1/` and the figures are in `thesis_final/images/`. None of the verifiers below needs them. What the bundle does contain:
- the clip index and the LoRA-adapted feature store for all 1,900 videos;
- the LoRA checkpoint, the normal prototypes and the teacher scores;
- the temporal test annotations;
- all Stage-2 run directories, including the three `dispatch_repro_*` checkpoints;
- the fixed evaluation bags and the per-run evaluations;
- the dense temporal outputs, the evidence summaries and the thesis figures.

The UCF-Crime videos (about 97 GB) are not redistributed. Download them from the dataset authors (Sultani, Chen and Shah, CVPR 2018; https://www.crcv.ucf.edu/projects/real-world/) and put them under `data/ucf_crime/raw/`, one folder per category (`Abuse/`, ..., `Vandalism/`, `Training_Normal_Videos_Anomaly/`, `Testing_Normal_Videos_Anomaly/`), as listed in `data/ucf_crime/splits/dman_v21/*.txt`. They are only needed for single-video inference and for re-running Stages 0 to 1b.

The frozen VideoMAE feature store (about 15 GB, `data/ucf_crime/artifacts/dman_v21/features/`) is only needed to re-run the pre-LoRA stage and the historical 14-class model, and to recompute its shard identity. It is available on request.

## Verifying the reported results

These checks need the evidence bundle, but no GPU and no raw videos. The verifiers recompute file hashes and metrics from the stored per-video outputs and compare them with the recorded evidence.

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

Each verifier ends with `PASS` or `"status": "pass"`. `--skip-raw-video-check` only skips the comparison of raw-video file sizes and modification times with the extraction records, which can match only on the original machine.

The prototype-distance study (called teacher ablation in the code) needs one extra step. `utils/verify_teacher_ablation_evidence.py` pins the hash of the trainer as it was when the ablation ran, and the trainer later gained the two normal-bag study options (off by default, see the notes below). So run the check against the recorded trainer, then restore the current one:

```bash
cp reproducibility/recorded_sources/m27_train_dispatch_mil.py stage2_class/src/train/
python -B utils/verify_teacher_ablation_evidence.py   # PASS: 12 runs, 84 artifacts, 48 recomputed metrics
git checkout stage2_class/src/train/m27_train_dispatch_mil.py
```

## Building the thesis

The sources are in `thesis_final/`. You need `lualatex` and `biber` (both in TeX Live), but not the evidence bundle.

```bash
cd thesis_final
latexmk -lualatex thesis.tex
```

If `latexmk` stops after creating the `chapters/` and `includes/` output folders, run it a second time. That happens when `-outdir` points to a new directory. The title-page date is the compile date, so build the PDF on the day it is handed in. The build is 93 pages without overfull boxes.

## Running the model on a UCF-Crime video

`inference/inference_any_video.py` runs a trained Stage-2 checkpoint on one UCF-Crime video. It prints the CAD-style alert and writes an annotated copy of the video. It needs the evidence bundle and the original video file at its place under `data/ucf_crime/raw/`, and takes a few seconds per video on CPU.

The example below reproduces the CAD figure in the thesis (seed 1337 checkpoint, test video `Abuse028`). The bundle already holds the thesis output for this video, so I copy it first and compare afterwards:

```bash
# needs data/ucf_crime/raw/Abuse/Abuse028_x264.mp4
cp artifacts/inference/dispatch_repro_s1337/Abuse028_x264_metadata.json /tmp/thesis_Abuse028_metadata.json

python inference/inference_any_video.py --video_id "Abuse/Abuse028_x264" \
    --run_tag dispatch_repro_s1337 --mode dispatch

diff /tmp/thesis_Abuse028_metadata.json artifacts/inference/dispatch_repro_s1337/Abuse028_x264_metadata.json \
    && echo "identical to the thesis output"
```

The expected alert is: anomaly detected, services POLICE and MEDICAL, priority P1-MEDICAL.

Outputs:
- per-clip model output (scores, attention, intervals): `artifacts/inference/<run_tag>/<video>_metadata.json`
- CAD-style alert record: `artifacts/inference/<run_tag>/<video>_alert.json`
- annotated video: `artifacts/visualizations/<run_tag>/<video>_dispatch.mp4`

Any other UCF-Crime video works the same way: pass `--video_id "<Category>/<file name without .mp4>"`, for example `"Explosion/Explosion017_x264"`. `--run_tag dispatch_repro_s2027` or `dispatch_repro_s3407` picks one of the other two official checkpoints. Re-running Abuse028 rewrites the bundled alert record for that video with a new timestamp, but the metadata stays the same.

The script uses the precomputed adapted features of the indexed video and checks the source file's frame count and frame rate against the clip index. This is the dense indexed system evaluated in the thesis. Videos outside UCF-Crime, or re-encoded copies, are not supported, because the thesis evaluates no raw-video-to-feature path for them. The CAD output is an offline research demonstration with no calibration, safety or operational validation.

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

Because the LoRA feature extraction used process-dependent sampling, a re-run produces a new feature store rather than the recorded one. Stage 2 results are therefore reproducible given the provided feature store.

## Notes on the recorded evidence

- The UCF-Crime test split was evaluated after model selection. The teacher ablation and the normal-bag study are validation only, and their configs refuse test evaluation.
- Run manifests and evidence records were written during development, so commit identifiers in them refer to my development repository. This repository is a clean snapshot of its final state. Some records also name their original folders (for example `RP_docs/evidence/` or `final_thesis_execution/phase_04_temporal_design_compute_preflight/`) and use internal phase numbers. The verifiers map those to `reproducibility/`, and I left the records unchanged so their hashes stay valid.
- `utils/extract_temporal_system.py` is byte-identical to the version recorded in the temporal evidence. To re-run it, set its `PHASE4` constant to `reproducibility/temporal_protocol`.
- `stage2_class/src/train/m27_train_dispatch_mil.py` gained the options `train.normal_response_loss` and `train.normal_response_weight` for the normal-bag study after the teacher ablation was recorded. Both are off by default and in every other config, so the training path of the official and ablation runs is unchanged. `reproducibility/recorded_sources/m27_train_dispatch_mil.py` is the earlier version, with SHA-256 `166c69ab…74cc9b` as pinned by the teacher-ablation evidence.
- `utils/verify_thesis_evidence.py` skips two inventoried files that are not part of this repository and prints the reason: `requirements.txt` was replaced by the pinned file, and a development report was not included.
