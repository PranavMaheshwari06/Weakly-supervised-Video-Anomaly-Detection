# Reproducibility Provenance Record

This file holds the artifact-identity hash tables and command transcripts that
supplement the thesis appendix "Reproducibility Record".
The appendix keeps the recorded-environment table and the determinism/limitations
section; this file holds everything else that identifies or reproduces a named
immutable input or output. SHA-256 is used for direct files; feature-store
identities hash the ordered list of shard paths and their individual SHA-256
values. Paths are relative to the repository root unless stated otherwise.

## 1. Dataset partition and clip index

| Artifact | SHA-256 |
|---|---|
| `data/ucf_crime/splits/dman_v21/train.txt` | `2503cb60061760475f2c81d690819e5c72bf3309f8954ac69bc5f195bb749db1` |
| `data/ucf_crime/splits/dman_v21/val.txt` | `d9a0edcbf3d38b1e4a43dc6db38b354ef22dbce21aaa28e93bdd4756223f1bcd` |
| `data/ucf_crime/splits/dman_v21/test.txt` | `afcb2a6bffe7286607e08c864425964d8784cbd5b1f263e03b76025e11548de7` |
| `data/ucf_crime/splits/dman_v21/stats.json` | `73912306500d72ae74c6893abe5f776e5eeca0033130319c99d37c6c7bbe9dcd` |
| `data/ucf_crime/artifacts/dman_v21/clip_index/videos.jsonl` | `dfb7fab0ef6dc5e155f1a14dbedd9b30d391a87dd9515f70fc77d0a7aae96bac` |
| `data/ucf_crime/artifacts/dman_v21/clip_index/stats.json` | `4556a802e7317564eedb8f7627815ee962297f5b3627d50d181b3753a3d488c4` |

The raw UCF-Crime video tree is approximately 97 GB and remains subject to its
access conditions; a dataset-wide raw-video hash was deliberately not computed.

## 2. LoRA adaptation checkpoint and sparse feature store

The selected LoRA checkpoint is `artifacts/stage1b/checkpoints/lora_best.pt`,
SHA-256 `2660d47b5a83f726039ec2eeb265a5a8cf08eac6d3f5dbd78ecbf552bc2d6df9`.
Its source configuration is `stage1b_finetune/configs/lora_finetune.yaml`,
SHA-256 `1bf1ed49e1ea759c13e6f653ef897887b4d4def7418b71439e687909a038e8c4`.

Standard LoRA feature-store identities (shard-list identity for the sparse
adapted feature store at `data/ucf_crime/artifacts/dman_v21/features_lora/`):

| Split | Feature-store identity SHA-256 | Shards |
|---|---|---|
| Train | `7a4fb455d9fe7741fee7bb99fb5fea437de7f4e30beca8912e6971c56bf31788` | 160 |
| Validation | `c0a8ed81ccfe39ab86468a2d20018ca13d642dd4ac77ec1fc70cc44ebb020c41` | 30 |
| Test | `96c4902ee26454a4facb405252342af309b03e3140252b3e4bdc742423d5cc36` | 49 |

## 3. Normal-prototype memory and teacher scores

The active response model consumes the LoRA-derived prototype artifact, not
the separate historical frozen-backbone prototype store.

| Artifact | SHA-256 |
|---|---|
| `artifacts/stage2/memory_lora/global_prototypes_k100.pt` | `d47f04d8a7184423473189a7669e6ff71f3bea9335aa7fb923c8c89883dbd31d` |
| `artifacts/stage2/teacher/teacher_scores_train.pt` | `83e606c383a6cc9d35a94a58ef0f5a6a6fb0767529da70aa9840d77eccbfab90` |
| `artifacts/stage2/teacher/teacher_scores_val.pt` | `0560aeb8ea7154cb172962e692333a9f25e82b838b57889f8f4077df59baa272` |
| `artifacts/stage2/teacher/teacher_scores_test.pt` | `0dbf85a9ce4f87dd4e2084276e50380fbbede9a538ab1018d6bf04c68b483111` |

The teacher directory above was subsequently overwritten in place with
LoRA-derived scores; the frozen originals are preserved alongside it as
`*.frozen_backup` files. The thesis's description of teacher-score construction
is historically correct for the executed run but is not reproducible from the
current configuration files alone.

## 4. Stage-2 configurations and checkpoints

The official Stage-2 source configuration is
`stage2_class/configs/hierA_dispatch_v1.yaml`, SHA-256
`03e10b129f85351a50691fd4d99eb1a73bf3666ae4fbfd2500a8b2a82ac1fa9f`. The fixed
validation and test manifests have direct file hashes
`a4c9174b3491ae302ecec9853c9d7ff526a0ded9b0aa98456bcf2ab409f2e67a` and
`9cddad6d40e9efe8f11c51982199ea1a2056a1c5be6131b4d4df083a37fc4041`. Portable
byte-identical copies are stored below
`reproducibility/evidence/dispatch_repro_v1/bag_manifests/`.

| Run | Resolved configuration SHA-256 | Selected checkpoint SHA-256 |
|---|---|---|
| `dispatch_repro_s1337` | `9ffd0272e7535e1626474be998b97d3147423c173b18a4ecad7e8b66cd653b56` | `95e04b0a1ca35ef859b6511c3eb86e706b24fa2e95126e8f528c16a3ef8a69c3` |
| `dispatch_repro_s2027` | `ecbcab202b0b091ff9cc70b6d7deb42fa41a2f3d84511ad7ebf71dcb19b483c3` | `7473fef046ac8cc8bfd9e4b5693844b8ecf113b7496a714303d4190f9d960bcb` |
| `dispatch_repro_s3407` | `48fc20d2a00d300c6c03dbf45de7cb63c542c83b8227af032d72338c7eaed0a9` | `6de4b613a013ae5c4f5e39845298c0b1801db9240d8685913a2ab9d5eba3fe16` |

The three checkpoints are evaluated separately.

## 5. Dense evaluation artifacts

The dense test feature manifest is
`data/ucf_crime/artifacts/dman_v21/features_lora_temporal_system_v1/manifest.json`,
SHA-256 `4385c610f53c8c1df041499e97d88202fc83ffbc7337ef08b3a0e5171477cb36`.
It lists 135 shards and 69,090 clips; the store identity is
`a2d1667f96c72a6918b87d932838fca00caca3b611f37e2fcdc7a664ea77ee7c`. Its
teacher tensor is
`artifacts/stage2/teacher_temporal_system_v1/teacher_scores_test.pt`, SHA-256
`398c53c42fdc242f012d475ac4c3433e6780c03b024ba90e933ca40ae99d32e5`.

## 6. Canonical evidence chain

- Fixed-bag summary: `reproducibility/evidence/dispatch_repro_v1/summary.json`,
  SHA-256 `ffb1e5d12ba1922a0e21e01eacab83ecd031a9ecc309e26df8a2ea0bea512b5c`.
- Validation-only teacher ablation:
  `artifacts/stage2/evidence/teacher_ablation_v1/summary.json`, SHA-256
  `edd6bda5065a1dea085a884f46ea5d81e79c0109796d9a266c1ee62819b89798`.
- Validation-only normal-bag response-loss study:
  `reproducibility/evidence/normal_bag_response_study/summary.json`, SHA-256
  `8123b0bec4e1da314b013fcb4e423e871b625231994501f2fd4ca3f71e9b6865`
  (Section 8).
- Dense temporal/system evaluation:
  `artifacts/stage2/evidence/temporal_system_v1/summary.json`, SHA-256
  `41f219f41474bdce58d0479e1d39e72a02811819564fbf4cee4d05d5031575d8`.
- Final quantitative authority:
  `artifacts/stage2/evidence/canonical_thesis_v1/package.json`, SHA-256
  `346ad82324c95db37fbbc92685c57ff316436f5237db29368196461e253a1a78`.

## 7. Figures

The 16 canonical vector figures under `artifacts/thesis_figures/phase6/` are
generated from the final package by `utils/generate_phase6_figures.py`,
SHA-256 `24264eedfb2cd2a6776c752d7fb07e9fd5c58f4f1f8429b60168e2f58e10234c`.
The vector figures included in the thesis have these hashes. The routing
comparison was first regenerated after a label correction and the architecture
figure after a font-size change. For print-size legibility, five
figures (architecture, protocol, routing comparison, temporal AP by category,
CAD timeline) were regenerated again from the same package with only figure
size, font size, and label wrapping changed; plotted values are unchanged. The
generator version for that regeneration has SHA-256
`7a3834bf90812ae04bd5e2e074cfe6f0fe8a0d74b93d7f29d85ab75ed7eeef0b`, and the
output is byte-identical across runs with `SOURCE_DATE_EPOCH=0`. The canonical
`artifacts/thesis_figures/phase6/` directory was not overwritten; the command
writes to a separate directory and the five PDFs were copied into the thesis
sources (`thesis_final/images/` in the development repository):

```
MPLCONFIGDIR=/tmp/mpl SOURCE_DATE_EPOCH=0 venv/bin/python -c "
import sys; from pathlib import Path
import utils.generate_phase6_figures as g
g.OUTPUT = Path('/tmp/r11_figures'); sys.argv = ['g', '--only'] + sys.argv[1:]; g.main()" \
  architecture provenance routing_comparison temporal_category cad_example
```

| File | SHA-256 |
|---|---|
| `thesis_final/images/fig_architecture_dataflow.pdf` | `0b169bc33fc1be465c065d34f025d11d1c21eebca813488f55d4339ac33cbf7f` |
| `thesis_final/images/fig_protocol_provenance.pdf` | `9e20b5e0fcd7d799aee7ee5d235c306eb92e49610f33cd7fa12830f63b355041` |
| `thesis_final/images/fig_joint_dispatch_distribution.pdf` | `76130875920b6e9e8cddd97943a09f474b7b2e4c46f07594b69121d7fb674666` |
| `thesis_final/images/fig_routing_comparison.pdf` | `93600d4246f5f1b7d9f48960141b7d098eef5d61da0f984d8399881875c3e610` |
| `thesis_final/images/fig_temporal_ap_by_category.pdf` | `545eebc7509d8c52909ac149c7dd0e8d9688d2c04bb827a6bee3f75e138ada3d` |
| `thesis_final/images/fig_temporal_cad_example.pdf` | `070cde351b0f536e40ec0dad3e5d477199e6599c23342b9df9714ca7641eac50` |

The final thesis no longer includes any file of this table (see the R19 and R20
sections below), and these files are not shipped in this repository. The hashes
are kept as the record of the earlier versions.

The rendered CAD-alert still frame added to Results
(`thesis_final/images/fig_cad_frame_example.png`) was produced from the
`dispatch_repro_s1337` checkpoint identified in Section 4 above, via
`inference/inference_any_video.py --video_id "Abuse/Abuse028_x264" --run_tag
dispatch_repro_s1337 --mode dispatch`, which writes
`artifacts/inference/dispatch_repro_s1337/Abuse028_x264_alert.json` (SHA-256
`82a377cff08c020641cdb20f20b7ecb0a2e6ab13e6b44ad9b90996a9712f5b5b`). The
figure was later re-rendered from the saved metadata of that same run with the
color-coded overlay (per-service colors, frame counter), without re-running
inference:

```
PYTHONPATH=. python3 -m stage3_dispatch.src.visualize_dispatch_video \
  --metadata artifacts/inference/dispatch_repro_s1337/Abuse028_x264_metadata.json \
  --video data/ucf_crime/raw/Abuse/Abuse028_x264.mp4 \
  --output artifacts/visualizations/cad_frame_colored_v1/Abuse028_x264_dispatch.mp4
```

Frame 75 (2.50 s) of that video is the figure, SHA-256
`cefcfd054b58ccdafb28c7c13c51614008034e47bb04b6055592b2ca9e92df7e`. Video, alert
record, and displayed values are unchanged; only the overlay styling differs
from the first rendering.

### R19 revision figures and bootstrap intervals

Figures 3.1 and 3.2 are now drawn in TikZ inside `thesis_final/chapters/methods.tex`
and contain no measured values; `fig_architecture_dataflow.pdf` and
`fig_protocol_provenance.pdf` are no longer included. Four figures were drawn
by `utils/generate_revision_figures.py` (SHA-256
`08c96acdbb013261e13c7c0b09f7cedb3778169a55d45a8e3cfcbeec063e7451`) from the
canonical package and the seed-1337 dense per-video records it lists (the
script checks each record hash). It writes only to the new directory
`artifacts/thesis_figures/revision_r19/`; the outputs were copied into
`thesis_final/images/` under the names below. The qualitative cases are chosen
by fixed rules in `select_cases()` (`--select-only` prints them).

```
python3 utils/generate_revision_figures.py
```

| File | SHA-256 |
|---|---|
| `thesis_final/images/fig_qualitative_cases.pdf` | `f10852a005e1d6c4e7751a983d024802d0376e97654591ef455e3c98455ff68f` |
| `thesis_final/images/fig_response_vector_distribution.pdf` | `01251eaf25ad526b6e7d91b4269130f975d221042b60c7619a838ada944678ec` |
| `thesis_final/images/fig_routing_comparison_r19.pdf` | `d51a1c8e4c52c90950523ad9bc2f2a6f0ade2f2d49af1551a07fcf6d298c7467` |
| `thesis_final/images/fig_temporal_ap_by_category_r19.pdf` | `e6878c9157ca4b51a659df7d234e78a48ab4ca00dde2e8fccfeccb2b6e5f49bc` |

These four R19 files were superseded by the R20 versions below and are not
shipped in this repository.

The paired video bootstrap intervals (B=10000, RNG seed 0, 95% percentile,
mean over the three seeds per resample) were computed by
`utils/bootstrap_test_cis.py` (SHA-256
`f627ab62ab0b97bdb4fcb6e9c7f10832b202227a3e536a70dce322fd9b35e8fe`) from the
same package and records. The script asserts that its point estimates
reproduce the reported values and refuses to overwrite its output
`artifacts/stage2/evidence/bootstrap_v1/bootstrap_cis.json`, mirrored to
`reproducibility/evidence/bootstrap_v1/bootstrap_cis.json` (SHA-256
`19b8daf25654f827a3dc796a4774d2007d8cc412df2ecea6bda7b42c7bbb4a9d`). Two
independent runs produced identical results.

```
python3 utils/bootstrap_test_cis.py
```

### Figures redrawn at print size (R20)

After the R19 draft, the figure labels were found to print below 8 pt. The five
figures listed below replace the R19 versions of the same plots in the thesis.
`utils/generate_revision_figures.py` was edited to draw them at the printed
width with larger fonts (SHA-256 of the edited script
`3831506adbe73b9436bc21be0e7ee5d28783daab152a35fbe0c20124468fd127`). The
routing comparison now has its two panels stacked, and the script also draws
the offline CAD timeline (previously drawn by the Phase-6 generator). The data
come from the same canonical package and the same seed-1337 dense records; the
selected cases and all plotted values are unchanged. The script writes only to
the new directory `artifacts/thesis_figures/revision_r20/` and refuses to
overwrite it; the R19 and Phase-6 directories were not changed.

```
python3 utils/generate_revision_figures.py
```

| File | SHA-256 |
|---|---|
| `thesis_final/images/fig_qualitative_cases_r20.pdf` | `340f860f1c277b365f64a6a168535ac1e6ada5793ee4efda16a7baa20bac4fc0` |
| `thesis_final/images/fig_response_vector_distribution_r20.pdf` | `e4133fffacad1451e35ad234050cee474795e1f29dd6f30fa496e4fd42967c4f` |
| `thesis_final/images/fig_routing_comparison_r20.pdf` | `410df183f9ec202b1bf9ed404493de7d75589c0851cadb6957c53ab18feac0cc` |
| `thesis_final/images/fig_temporal_ap_by_category_r20.pdf` | `76e3acf4e57e0049d89da9d0523cae5da4f1d2c310ec486d78a26302f6fdbdb6` |
| `thesis_final/images/fig_temporal_cad_example_r20.pdf` | `81c5e1fdfab702150fbd54564db4753fb95039c593698a38978181813839d695` |

Re-running the script reproduces the plotted content but not the file hashes:
matplotlib writes the creation time into every PDF. For the two files compared
(qualitative cases and CAD timeline) the regenerated PDFs differ from the
listed ones only in the 5 bytes of that time stamp. The hashes above identify
the files included in the thesis.

The title page now shows the chair line and the compile date, the declaration
is the unmodified template text, and an acknowledgements page was added. None
of these contain measured values. The R19 files listed above and the earlier
figure files remain in the development repository's `thesis_final/images/` but
are no longer included in the thesis and are not shipped here. This repository
ships `thesis_final/images/` with the five R20 PDFs, `fig_cad_frame_example.png`
and `uni-logo.png`.

## 8. Normal-bag response-loss study (validation only)

A post-hoc mechanism study run after the official protocol was frozen. It is
not part of the official evidence tier and none of its checkpoints was
evaluated on the test split: each resolved configuration sets
`reproducibility.validation_only: true`, which `m29_eval_dispatch` enforces, and
each run manifest records `test: embargoed_not_loaded`. The control arm is
`ablation_full_s{1337,2027,3407}` from the validation-only teacher ablation
(Section 6). All arms use the fixed validation manifest `a4c9174b…e67a`
(Section 4). The per-arm aggregate values are mirrored in
`reproducibility/evidence/normal_bag_response_study/summary.json`, SHA-256
`8123b0bec4e1da314b013fcb4e423e871b625231994501f2fd4ca3f71e9b6865`.

| Source configuration | SHA-256 |
|---|---|
| `stage2_class/configs/abl_normal_response_v1.yaml` (unweighted) | `ac4a77c7fe68a5e18e43dfa8cbf8a4a5cbb284731e9f6f7ad568a8c448072023` |
| `stage2_class/configs/abl_normal_response_reweighted_v1.yaml` (reweighted, weight 0.3) | `8094831aafe27ea65aaf92b32718467f40f1a489b5ce14723acf78262fab914f` |

| Run | Resolved configuration | Selected checkpoint | Validation evaluation |
|---|---|---|---|
| `abl_normal_response_v1_s1337` | `b35b87a723646da3e0a9b670d18a5a379854f4de347eee52179150d650444cd6` | `dfa5f22ddcb89532dd21d7367ad638c6deec7f31fcc373116ccdfee80e545bed` | `e1fd0bdbd1edeb2376e94417035dc8503006eef8101c9be4bec4266ab1fcba66` |
| `abl_normal_response_v1_s2027` | `15226ea89c2335151dff981ef99417db0c81818c9677d3b7bac45719aa1b2c3a` | `050141b0a80490c69905de6293e199b788502b7a412e217b2465d62ae379cc00` | `73894fe7f6c0cfde7b344674c5be86e579ef9963ea54993c522468ee5c8cd6f1` |
| `abl_normal_response_v1_s3407` | `05f7c7e8c1ac7aa12b1a9e2a77ad3e1e9e397aea4b50ce4d72226a5248d3c156` | `ec402471569d1723fc31e48025178db9bb6cccb3b11034922e92604d9aaf755d` | `f4969011f6c5230d4924a1182cfb3d7465cee88e5914f2132c05795905659fd5` |
| `abl_normal_response_reweighted_v1_s1337` | `05566078e156149aec2179b856e3210c3119029545740cbefbd79f1b5a3cda58` | `81ebe802939d6c42abb098c8a4b95ff67a8160229c56949ff5c0066902c7bd8e` | `8ad1aa549091d42cc064e9eec97bb2e06463ac66be0fb1e1f5ec0976b0f86d4c` |
| `abl_normal_response_reweighted_v1_s2027` | `3800d95d7993221aedb7c1bb21ff84f0896b46edd37e3ce3ee3168227457aac7` | `7182c0f87b5abde4e782922e52291860e50c6009fe891ae16e16b22a28599401` | `641a7440d3b324435ae1f6ffcee7f9b639bbe3b1b33e8f46737b288287127b6c` |
| `abl_normal_response_reweighted_v1_s3407` | `6ab07b1c12541f3a99f8fc36b0dfdbe83e1f3355fd7294c392735ed1a3329cda` | `183d6c58eb92524c496ec1f6370d8b1f6925d8b5c0f313634d018ee5452b4495` | `88e662b1c6b610e8a34c6980359a408feaa45baf6fd1c20aa971372fecefa016` |

Checkpoints are `artifacts/stage2/runs/<run>/checkpoints/mil_best.pt`;
evaluations are `artifacts/stage2/eval/<run>/dispatch_eval_val.json`. The run
manifests record a development-repository commit with uncommitted changes; the
`train.normal_response_loss` and `train.normal_response_weight` flags were
committed afterwards, and both source-configuration hashes above match the
files in this repository. Command form, per seed:

```
PYTHONPATH=. python3 stage2_class/src/train/m27_train_dispatch_mil.py \
  --cfg stage2_class/configs/<CONFIG>.yaml --run_tag <RUN_TAG> --seed <SEED>
STAGE2_CFG=stage2_class/configs/<CONFIG>.yaml \
python -m stage2_class.src.eval.m29_eval_dispatch --run_tag <RUN_TAG> --split val
```

## Command Record

Commands below assume the repository root as the working directory. They are
separated by purpose because no single shell script reproduces the final
evidence from raw video to all reports. Historical Stage-1b execution did not
save an exact command transcript or full environment manifest; the first
group therefore records the source-defined entry points and immutable
selected output, not a claim of a contemporaneous shell log.

### Adaptation training entry point

```
venv/bin/python -m stage1b_finetune.src.train_lora \
  --cfg stage1b_finetune/configs/lora_finetune.yaml
```

### Sparse feature extraction entry points

```
venv/bin/python -m stage1b_finetune.src.extract_lora_features \
  --cfg stage1b_finetune/configs/lora_finetune.yaml \
  --lora_ckpt artifacts/stage1b/checkpoints/lora_best.pt --split train
venv/bin/python -m stage1b_finetune.src.extract_lora_features \
  --cfg stage1b_finetune/configs/lora_finetune.yaml \
  --lora_ckpt artifacts/stage1b/checkpoints/lora_best.pt --split val
venv/bin/python -m stage1b_finetune.src.extract_lora_features \
  --cfg stage1b_finetune/configs/lora_finetune.yaml \
  --lora_ckpt artifacts/stage1b/checkpoints/lora_best.pt --split test
```

These commands must not overwrite the immutable stores named above.
Re-execution requires new output locations and an explicit identity
comparison. Because sparse subset selection used process-dependent Python
hashing, a configuration-only rerun is not expected to recreate the official
store identity automatically.

### Stage-2 training and fixed-bag evaluation

The following command form was used for each official seed. The recorded run-tag/seed pairs are:

- `dispatch_repro_s1337` with seed 1337;
- `dispatch_repro_s2027` with seed 2027; and
- `dispatch_repro_s3407` with seed 3407.

Substituting each pair gives the executed training commands. The validation
command was used for checkpoint selection; test evaluation was run only after
selection was frozen.

#### Training

```
STAGE2_CFG=stage2_class/configs/hierA_dispatch_v1.yaml \
python -m stage2_class.src.train.m27_train_dispatch_mil \
  --run_tag <RUN_TAG> --seed <SEED>
```

#### Fixed-bag validation and test evaluation

```
STAGE2_CFG=stage2_class/configs/hierA_dispatch_v1.yaml \
python -m stage2_class.src.eval.m29_eval_dispatch \
  --run_tag <RUN_TAG> --split val

STAGE2_CFG=stage2_class/configs/hierA_dispatch_v1.yaml \
python -m stage2_class.src.eval.m29_eval_dispatch \
  --run_tag <RUN_TAG> --split test
```

The existing run and evaluation directories are protected against overwrite.
The validation-only ablation's exact 12 training and 12 evaluation commands
are embedded with every run artifact hash in
`artifacts/stage2/evidence/teacher_ablation_v1/summary.json`. Its results can
be checked without retraining:

```
python3 -B utils/verify_teacher_ablation_evidence.py
```

### Dense extraction

The final dense extraction used the following commands:

```
venv/bin/python -m utils.extract_temporal_system
venv/bin/python -u -m utils.resume_temporal_extraction
```

### Dense indexed evaluation and release verification

```
venv/bin/python -u -m utils.evaluate_temporal_system \
  --device cuda --attempt-id attempt_002
venv/bin/python -u -m utils.verify_temporal_system \
  --attempt-id attempt_002 --release
```

The first extraction process was externally interrupted after writing a
verified prefix. The second command resumed from that exact prefix under the
unchanged protocol. Attempt 001 remains immutable and incomplete; attempt 002
is the released evidence source. This is technical recovery, not model or
threshold selection.

### Evidence verification and figure generation

The principal non-training checks are:

```
python3 -B utils/verify_thesis_evidence.py
python3 -B utils/verify_historical_14_class_evidence.py
python3 -B utils/verify_teacher_ablation_evidence.py
venv/bin/python -B -m utils.verify_temporal_preflight
venv/bin/python -B -m utils.verify_temporal_system --attempt-id attempt_002
venv/bin/python -B -m utils.verify_canonical_evidence_package \
  --require-figures
```

The canonical package and figures were created by deterministic
post-processing of existing evidence, not by model inference:

```
venv/bin/python -m utils.build_canonical_evidence_package
MPLCONFIGDIR=/tmp/phase6_matplotlib \
  venv/bin/python -m utils.generate_phase6_figures \
  --preview-dir /tmp/phase6_previews_final
```
