**UCF-Crime Anomaly Detection | DMAN v2.1 Thesis Pipeline**
# Stage 0 — Data Plumbing & Preprocessing  


## 1. Purpose of Stage 0

Stage 0 establishes a **clean, reproducible, and verifiable data pipeline** for video anomaly detection using the **UCF-Crime (Anomaly Detection experiment)** dataset.

The goal of this stage is **not model training**, but to ensure that:
- raw videos are correctly organized and validated,
- train / validation / test splits are well-defined and leak-free,
- videos are decoded in a consistent temporal format,
- fixed-length clips are explicitly defined,
- and model-ready tensors can be loaded end-to-end without ambiguity.

All later stages of the thesis pipeline (feature extraction, memory modeling, MIL training, evaluation) rely on the **data contract** established here.

---

## 2. Dataset Origin and Experimental Setting

### 2.1 Dataset Variant Used

This project uses the **UCF-Crime Anomaly Detection experiment**, which consists of:
- 13 anomaly event categories
- normal videos for training and testing
- official train/test split provided by the dataset authors

This stage does **not** use the Action Recognition experiment splits.

---

### 2.2 Original Dataset Distribution

The dataset is originally released as multiple zip archives:

#### Anomaly Detection Videos
- `Anomaly-Videos-Part-1.zip` … `Anomaly-Videos-Part-4.zip`
- `Training-Normal-Videos-Part-1.zip`
- `Training-Normal-Videos-Part-2.zip`
- `Testing_Normal_Videos.zip`
- 

#### Official Split Definitions
- `UCF_Crimes-Train-Test-Split.zip`
  - contains folder `Anomaly_Detection_splits/`
  - includes:
    - `Anomaly_Train.txt`
    - `Anomaly_Test.txt`

#### Temporal Annotations (Testing Only)
- `Temporal_Anomaly_Annotation_For_Testing_Videos.txt`

> Temporal annotations are preserved for later analysis but are **not required for Stage 0 preprocessing**.

---

## 3. Dataset Setup in WSL

All zip files were extracted once and organized into a **stable folder layout** inside WSL.

Final dataset root used by the pipeline:
    data/ucf_crime/
    ├── raw/
    │ ├── Abuse/
    │ ├── Arrest/
    │ ├── Arson/
    │ ├── Assault/
    │ ├── Burglary/
    │ ├── Explosion/
    │ ├── Fighting/
    │ ├── RoadAccidents/
    │ ├── Robbery/
    │ ├── Shooting/
    │ ├── Shoplifting/
    │ ├── Stealing/
    │ ├── Vandalism/
    │ ├── Training_Normal_Videos_Anomaly/
    │ └── Testing_Normal_Videos_Anomaly/
    ├── annotations/
    │ └── Temporal_Anomaly_Annotation_For_Testing_Videos/
    └── splits/
    └── Anomaly_Detection_splits/
    ├── Anomaly_Train.txt
    └── Anomaly_Test.txt


This layout mirrors the dataset semantics and avoids any implicit assumptions.

## 4. Overview of Stage 0 Pipeline

Stage 0 is divided into **five focused sub-stages**, each validated independently:

    Raw Videos
    ↓
    Stage 0.1 — Dataset Indexing
    ↓
    Stage 0.2 — Train / Validation / Test Split Construction
    ↓
    Stage 0.3 — Video Decoding
    ↓
    Stage 0.4 — Clip Index Construction
    ↓
    Stage 0.5 — End-to-End Dataloader Sanity Check



Each stage produces explicit artifacts reused downstream.

---

## 5. Stage 0.1 — Dataset Indexing

**File:** `ucf_paths.py`

### Purpose
- Validate that all expected folders exist
- Enumerate `.mp4` files deterministically
- Associate folder names with semantic labels
- Verify dataset integrity once, upfront

### Outcome
- Total indexed videos: **1900**
- Folder-based grouping used consistently across the pipeline

This prevents silent mismatches between dataset structure and later training logic.

---

## 6. Stage 0.2 — Train / Validation / Test Split Construction

**File:** `split_builder.py`

### Why This Step Is Needed
The dataset provides an official **train/test split**, but a **validation set** is required for:
- hyperparameter tuning,
- checkpoint selection,
- avoiding test-set leakage.

### Logic
1. Read official split files:
   - `Anomaly_Train.txt`
   - `Anomaly_Test.txt`
2. Verify that all referenced videos exist under `raw/`
3. Ensure no overlap between train and test
4. Create validation split:
   - 10% of the official training set
   - stratified by top-level folder (event category / normal)
   - deterministic random seed

### Output

    data/ucf_crime/splits/dman_v21/
    ├── train.txt (1449 videos)
    ├── val.txt (161 videos)
    ├── test.txt (290 videos)
    └── stats.json



Final counts:
- Train + Val: **1610**
- Test: **290**
- Total: **1900**

---

## 7. Stage 0.3 — Video Decoding

**File:** `video_decode.py`

### Purpose
Define a **single, consistent decoding procedure** for all later stages.

### Design Decisions
- Use OpenCV for robust baseline decoding
- Convert videos to a **fixed temporal resolution**:
  - target FPS = **15**
- Output format:
  - RGB
  - `torch.uint8`
  - shape `(T, H, W, 3)`

### Why 15 FPS?
- UCF-Crime videos are typically ~30 FPS
- Downsampling reduces redundancy
- Fixed temporal scale is standard in video representation learning
- Enables consistent clip duration across videos

A separate sanity script verifies decoding correctness before proceeding.

---

## 8. Stage 0.4 — Clip Index Construction

**File:** `clip_index.py`

### Why This Step Exists
The learning pipeline operates on **clips**, not full videos.

For Multiple Instance Learning (MIL) and memory-based anomaly detection, each video must be treated as a **bag of fixed-length instances**.

### Clip Definition
- Temporal timeline: **15 FPS**
- Clip length: **16 frames** (~1.07 seconds)
- Stride: **8 frames**

### What Is Built
For each video:
- number of frames on the 15 FPS timeline
- list of valid clip start indices
- metadata required for deterministic sampling

### Output
    data/ucf_crime/artifacts/dman_v21/clip_index/
    ├── videos.jsonl
    └── stats.json


Observed scale:
- Videos indexed: **1900**
- Candidate clips: **857,873**

This index is a **superset**. Later stages sample from it; they do not load all clips.

---

## 9. Stage 0.5 — End-to-End Dataloader Sanity Check

**Files:**
- `src/data_loader.py` (the one-off check script is not part of this repository)

### Purpose
Prove that the entire preprocessing pipeline works end-to-end.

### What Is Verified
- split files load correctly
- clip indices align with decoded frames
- clip slicing is valid
- spatial transforms are correct
- final tensors match model expectations

### Final Clip Format
    shape: (3, 16, 224, 224)
    dtype: torch.float32
    range: [0, 1]





Only a few clips are sampled. This stage is a **sanity gate**, not a training step.

---

## 10. Why Stage 0 Is Critical

Stage 0 ensures:
- no hidden data leakage,
- no silent preprocessing bugs,
- reproducibility across experiments,
- a clear and inspectable data contract.

While many papers assume these steps implicitly, this thesis **explicitly implements and validates them**.

---

## 11. Relation to Later Stages

- Stage 1 uses clip-level tensors for anomaly scoring and memory modeling
- Stage 2 uses clip inventories for MIL training and sampling
- Evaluation and ablation rely on fixed splits and clip definitions

Stage 0 is foundational and is **not reimplemented** later.

---

## 12. Stage 0 Completion Checklist

Stage 0 is complete when:
- split counts match expected totals (1449 / 161 / 290)
- decoding sanity passes
- clip index is built for all 1900 videos
- dataloader sanity returns valid `(3,16,224,224)` tensors

---

## 13. Summary

Stage 0 converts raw UCF-Crime videos into a **reproducible, clip-based dataset** suitable for anomaly detection research.  
It establishes the data integrity and preprocessing assumptions upon which the entire thesis pipeline is built.

**Stage 0 is complete and locked.**
