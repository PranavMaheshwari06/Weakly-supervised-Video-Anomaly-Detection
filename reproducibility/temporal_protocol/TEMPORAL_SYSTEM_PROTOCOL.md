# Frozen Phase 4 temporal and indexed-system protocol

Protocol ID: `temporal_system_v1`. Frozen on 2026-09-19 before any official
temporal model evaluation. `protocol.json` records exact inputs, decisions and
output locations; `coverage_audit.json` records the complete per-video inventory.
This document defines the metric semantics. Neither file may be revised in
response to test predictions. An implementation defect stops execution and is
reported before any revised protocol is considered.

## Checkpoints, question and evaluation population

RQ4 uses **all three** official checkpoints: `dispatch_repro_s1337`,
`dispatch_repro_s2027`, `dispatch_repro_s3407`, at their existing validation-selected
epochs 34, 0 and 25. Each is `artifacts/stage2/runs/<tag>/checkpoints/mil_best.pt`.
Exact SHA-256 values, resolved configs and original run manifests are in
`protocol.json`. No seed, epoch or ablation variant is selected using temporal
annotations, prior test results, or the Phase 3 ranking. No ensemble is substituted
for per-checkpoint evaluation. Phase 3 remains a validation-selected, n=3 mechanism
study, with no test/generalization or operational evidence.

The population is the existing ordered `splits/dman_v21/test.txt`: 290 videos,
140 anomalous and 150 normal. Every video receives a result/status row for every
checkpoint. The primary temporal denominator is the **135** anomaly videos with
valid positive annotations within source bounds. Normal videos have genuinely
empty annotations, not missing annotations. All 290 remain in indexed routing.

Five videos are excluded from primary temporal metrics because annotation ends
exceed source length; their original bounds and reasons remain in the audit:

| Video | Annotation end | Source frames |
| --- | ---: | ---: |
| Arson/Arson011_x264.mp4 | 1267 | 1266 |
| Arson/Arson016_x264.mp4 | 1796 | 1795 |
| Explosion/Explosion033_x264.mp4 | 3156 | 3154 |
| Fighting/Fighting003_x264.mp4 | 3103 | 3102 |
| Shooting/Shooting015_x264.mp4 | 1715 | 1713 |

Do not silently clip these annotations, turn them into normal videos, select only
successfully localized videos, or infer annotations from class names. The local
text and MATLAB annotations agree; both preserve the same boundary conflicts.
Temporal exclusion does not remove a video from routing or coverage accounting.
No post-hoc corrected-ground-truth sensitivity run is authorized by this protocol.

## Source-frame and time contract

Use indexed `src_fps`, `keep_every_n`, `src_frame_count`, `clip_len=16`, `stride=8`.
Validate positive finite FPS, integer positive counts, and
`keep_every_n=max(1, round(src_fps/15))`, matching the decoder's Python rounding.
Every current test file was opened and its FPS/count checked against the index.
Every current test video is 30 FPS, 320×240 with decimation 2, but the conversion
must also work at other source rates. The effective sample rate is `src_fps/n`.

All evaluation intervals are **zero-based half-open** `[start,end)` source-frame
intervals. Sampled start `s` represents source samples `(s+j)*n`, j=0..15.
Its temporal support is `[s*n, min((s+16)*n,N))`: each sampled frame represents
its decimation cell. This is temporal support, not a claim that every source
frame was fed to VideoMAE. Reject a clip if `(s+15)*n >= N`; do not pad it.
Source-frame boundary time is `frame/src_fps`. No `fps/15` approximate mapping,
fixed `2.0`, guessed FPS, `N+1` mask, or fixed 30-FPS seconds conversion is allowed.

Positive annotation frame numbers are interpreted as **1-based inclusive**:
`[a,b] -> [a-1,b)`. The README describes start/end frame numbers and the MATLAB
companion, but does not explicitly name an index origin. This is a disclosed
frozen convention, supported by positive bounds (including 1), not a claim of
explicit documentation. `(-1,-1)` is an absent event; partially negative,
reversed or out-of-range pairs are invalid. Merge overlaps before measuring.

The existing renderer/CAD metadata expects zero-based **inclusive** ends. Adapt
`[a,b)` to `{start_frame:a,end_frame:b-1}` only at that boundary. Its end time is
`(end_frame+1)/fps`. `alert_latency_sec` is a legacy name for video-relative
trigger time, never computation latency. No event onset error or operational
response-time claim is part of this evaluation.

Implementation authority: `stage2_class/src/eval/temporal_geometry.py`, used by
the corrected indexed helper and covered by targeted tests. Historical m28/m28b
outputs and figures retain their original semantics and are not official inputs.

## Dense features and compute

Select the **entire existing candidate grid**, without changing clip starts,
stride, thresholds or source data. There are 69,090 candidates across 290 videos.
The existing sparse store has 27,516 clips. Candidate-grid coverage is 99.8431%
of source frames versus 50.8956% for sparse features and 12.0624% for fixed bags.
These are frame-weighted support fractions; video means are different and also
recorded. Residual uncovered tails stay uncovered. Candidate-grid coverage is a
geometric measurement, not proof all future clips will successfully decode.

Extract all candidates into the new store, including candidates already present
in the sparse store. Never mix old and new embeddings opportunistically. Do not
use `deterministic_sample_starts`, Python's process-dependent hash, a 128-clip cap,
or the old extractor's full-video tensor accumulation. Retain at most one batch
of four transformed clips plus a bounded output shard buffer (512 clips).
Write float16 `[M,768]` embeddings and exact keys/start/source metadata. Preserve
the existing decode, shorter-side resize/center crop, processor normalization,
mean-token pooling, LoRA checkpoint, and no-gradient fp16-autocast behavior.

Use local RTX 4050 Laptop GPU compute. The benchmark measures 297 validation clips
twice, from shortest/median/longest normal and anomaly videos, with four spread
contiguous windows per video. All have the same 320×240 resolution as test. The
long normal sample is 20,934 seconds, beyond the test maximum of 3,599.9 seconds;
its seek overhead gives a conservative sampled rate. Two warm-up batches precede
timing. Peak allocated/reserved memory is 586,489,344 / 780,140,544 bytes, against
6,438,780,928 bytes dedicated VRAM. Shared RAM is not VRAM. Projection is 6,587.9
seconds at the weighted rate or 11,187.3 seconds at the slowest sample rate.
Allow a **four-hour extraction planning window**, plus scoring/evaluation and I/O;
this is an estimate, not a deadline or latency guarantee. Payload is about 106 MB
before keys/metadata. No additional/cloud compute is needed on current evidence.

The benchmark is decode/transform/transfer/forward/CPU-return timing; features
were discarded. It excludes final shard writes, teacher scoring, MIL evaluation,
video rendering, thermal/load variation over a full run and external CAD work.
It does not establish live or multi-stream throughput. On OOM or decode failure,
record the failure and stop; do not silently fall back to CPU or change batch size.

## Backbone compatibility and teacher inputs

Use cached `MCG-NJU/videomae-base` revision
`dc740ceda42fce44faed2ea03c6d447db72f6af9`, current installed package versions in
`annotation_environment_audit.json`, and LoRA checkpoint SHA-256 in the manifest.
Offline loading only. Hash the cached model/config/processor against
`backbone_replay.json`. The current Transformers 5.12.0 loader reports missing
Linear query/key/value biases and ignores the cached old q_bias/v_bias names.
This behavior is a methodological limitation and must be disclosed in Phase 7.

Validation feature replay of 18 first/middle/last stored clips from the six
benchmark videos found current-loader max absolute error 0.0009765625 and minimum
cosine similarity 0.9999997616. Restoring the published biases changed features
substantially (max error 3.1484375; min cosine 0.8732984). That alternative is
**rejected for compatibility**, not evaluated for task performance. Do not repair
the published-weight loading semantics mid-experiment. Dense features must retain
the existing stored representation. No retraining or prototype refitting occurs.

Before full extraction in Phase 5, replay the same validation keys in a new
immutable preflight record, requiring finite features, max absolute error ≤0.005
and cosine ≥0.99999 for each clip. These are engineering compatibility tolerances,
not model-selection criteria. Failure blocks extraction. Verify every adapter key
loads and verify newly initialized attention biases are zero. Do not select a new
loader or checkpoint based on test results. Record actual loaded model state
identity for the new extraction run; do not retroactively invent historical state.

Teacher scores use the existing **LoRA** normal prototypes at
`artifacts/stage2/memory_lora/global_prototypes_k100.pt`, not the frozen-backbone
memory at `memory/`. Existing active teacher files identify `memory_lora` in their
metadata. Keep the persisted float16 prototype values cast to float32, with no
refit or renormalization of prototype rows; L2-normalize each float16-on-disk
feature in float32 (epsilon 1e-12), then compute `min_k ||x-p_k||²` with the existing
dot-product formula. No score scaling, clipping or min-max calibration is used
for MIL input. Append the scalar to the original unnormalized feature (769 dims).
Saved prototypes are rounded from the original training-time float32 centers;
the original centers are unavailable. Record validation replay discrepancy against
stored teacher scores, require absolute error ≤0.001, and disclose this finite
precision limitation. Failure blocks extraction/evaluation, not a new memory fit.

## Forward pass and distinction between protocols

The final **dense indexed-system** uses every successfully verified candidate,
ordered by `(clip_start,clip_key)`, in one variable-length video bag with a true
mask and no resampling, truncation, padding or fixed-bag substitution. Fail on
duplicate keys, missing teacher scores, nonfinite tensors, wrong dimensions,
unexpected checkpoint keys or metadata/config mismatch. Use the original
`HierarchicalAttentionMilHead` (769 input, 256 attention hidden, 3 channels,
top-quarter pooling `max(1,int(M*0.25))`), eval mode, no MIL autocast, float32.
Preserve the existing Torch top-k behavior and record environment/device. Exact
video IDs are required; the helper's substring search is not permitted for batch
evaluation. The Phase 5 evaluator must implement this protocol explicitly.

The old indexed helper currently loads the sparse store; its new geometry does
not by itself make it an official dense evaluator. The frozen dense store and
teacher paths must be passed explicitly by Phase 5. This is a declared input
coverage change, not a silent relabeling of past sparse demonstrations.

The original **fixed-16-bag** results remain exactly as published under their
immutable bag manifest, feature store and teacher files. Do not rerun or overwrite
them with dense inputs. Their probabilities can be read later for matched raw,
conditional/gated comparisons in Phase 6. No numeric result here is a new official
temporal or indexed-system result. No sparse-versus-dense task-performance gain
claim follows just from the geometric coverage audit.

## Temporal metrics and baselines

Primary metrics measure ungated attention localization on the 135 eligible
anomaly videos. A binary false negative does not remove a video. Per-checkpoint
raw attention is the shared MIL softmax attention; it is not a calibrated anomaly
probability. Store it at full precision with the ordered keys and support.

1. **Source-frame AP:** project raw clip attention to source support using the
   maximum of overlapping clip scores. Uncovered source frames receive a score
   strictly below the minimum clip score (`min_score-1`). Ground truth is a mask
   of length N. Compute noninterpolated AP as `sum(delta_recall * precision)` at
   distinct score thresholds, grouping tied scores; equivalently sklearn
   `average_precision_score`. Compute AP separately per video, then average over
   eligible videos. Do not pool cross-video attention into a primary micro AP:
   attention normalization depends on bag length. Constant scores yield anomaly
   prevalence. Normal/missing-label AP is null and excluded, not zero.
2. **Top-attended-clip hit:** select greatest raw attention, breaking exact ties
   by earliest start then clip key. Hit iff its support has positive intersection
   with any annotated interval. Record overlap frames and chosen key. The video
   hit rate is the mean of these indicators; no binary gating or minimum IoU.
3. **Fixed-rule IoU:** smooth attention with Gaussian sigma=2, reflect boundaries,
   separately inside runs whose starts differ exactly by stride 8. Min-max
   normalize the resulting scores once over the whole video. Constant vectors
   map to zero. Select scores **≥0.5** and union their source supports. Do not fill
   gaps between disjoint supports. IoU is union-level `|P∩G|/|P∪G|`; empty P on an
   anomalous video is zero. Report mean/median IoU, fraction ≥0.1/0.3/0.5/0.7,
   exact-zero count and frequency. No sweep or threshold optimization.
4. **Coverage:** for fixed-bag, sparse and realized dense clips separately, union
   support/N, max uncovered gap, clip/unique counts, and annotated-support recall
   `|C∩G|/|G|`. Report per-video values, video mean/median/minimum and frame-weighted
   totals. Actual extraction failures must not be concealed by candidate coverage.

Mandatory matched temporal baselines on the same eligible set and support:

- **Whole video:** frame score constant 1, P=[0,N), AP=prevalence, IoU=prevalence;
  top-clip hit is null because this baseline has no chosen clip.
- **Uniform clips:** every clip gets 1/M; identical projection and interval rules,
  earliest clip tie-break for top hit. Constant-vector normalization yields no
  predicted interval, by design, and exposes that edge case.
- **Seeded random clips:** one draw per video, independent of checkpoint. NumPy
  `Generator(PCG64(int.from_bytes(sha256(('temporal_system_v1:'+video_id).encode()).digest()[:8],'big')))`;
  sample M Uniform[0,1) numbers, divide by their sum, same projection/interval rules.
  No best-of-seeds or selection of favorable draws. Label as a single random
  reference, not a random-baseline confidence interval.
- **Teacher distance:** use the same dense teacher scores as clip saliency, same
  projection, top-hit and smoothing/interval rules; no MIL or new fitted threshold.

Baselines are computed once and are seed-independent; never count three copies as
three independent baseline runs. All temporal metrics are also broken down by
category with support counts; the primary macro unit remains video, not category.

## Exact indexed routing and CAD rules

For each of 290 videos and each checkpoint, save binary logits/softmax anomaly
score and all three raw dispatch logits/sigmoids in Police, Medical, Fire order.
Binary prediction is argmax of the two logits (tie -> Normal). Primary raw
dispatch threshold is **0.5** independently per channel. Primary gated vector is
zero when binary predicts Normal, otherwise the thresholded vector. Separately
evaluate the existing **demo** thresholds `[0.65,0.60,0.65]` with the same gate.
No temporal interval is required to trigger the video-level binary gate.

For each threshold vector report raw, binary-gated and anomaly-conditional
(ground-truth anomalous 140 videos, not predicted-positive) routing. Report
per-channel TP/FP/FN/TN, precision/recall/F1, three-channel Macro-F1, Hamming loss,
exact match and predicted-vector counts. Undefined precision/recall/F1 denominators
give zero, with supports shown. Binary metrics on all 290: AUROC, F1, accuracy,
precision/recall and confusion. Targets are the unchanged project-derived mapping
in `dispatch_label_map.py`; unknown anomaly groups fail rather than default to
Normal. Mandatory routing baselines are all-services `[1,1,1]` and the **same
checkpoint's binary-gated all-services** vector. They use identical video sets.

CAD alerts use demo thresholds and `generate_alert`: binary Normal -> no alert,
empty services and CLEAR; otherwise Fire -> P1-EMERGENCY, else Medical ->
P1-MEDICAL, else Police -> P2-POLICE, else P3-REVIEW. Generate intervals by the
same smoothing/union contract with fixed **0.65** display threshold. Record both
ungated display intervals and effective gated intervals. For anomaly without an
interval, keep triggered/P3-or-service status, null anomaly window and trigger
time; do not invent frame-zero detection. The router's timestamp is presentation
metadata, excluded from deterministic comparisons. Record service vector and
priority; no external notification is sent.

Report secondary **gated display IoU/zero-IoU** on the same 135 eligible anomaly
videos using display threshold 0.65 and empty P for binary false negatives. Keep
these separate from primary ungated 0.5 IoU. For the 150 normal videos report
predicted temporal-support fraction and any-interval frequency (ungated 0.5 and
gated display 0.65); do not define empty/empty IoU as a localization success.

## Aggregation, failures and immutable outputs

For each checkpoint, aggregate eligible per-video values first. Then report all
three checkpoint values and arithmetic mean ± sample SD (`ddof=1`, n=3), with no
inferential significance or generalization claim. Keep the 135-video temporal,
150-video normal, 140-video conditional-routing and 290-video routing denominators
explicit. No pooling of 870 seed-video rows as independent observations. AP/top-hit
and IoU have the same eligible anomaly denominator; category summaries state
counts. A metric with no eligible observations is null with reason, never NaN.

Record failures with `video_id`, `checkpoint` (nullable before MIL), stage, code,
message, affected keys, expected/observed counts and retry identity. Codes include
missing/ambiguous metadata, annotation_missing/invalid/out_of_bounds, source drift,
decode_failure, feature/teacher_missing, duplicate_key, nonfinite, shape_mismatch,
checkpoint_mismatch, OOM and output_collision. Annotation exclusions are already
fixed; any new extraction/inference failure blocks official completion. Retain
all expected rows with null unavailable metrics; do not silently shrink the set,
substitute zeros for failed predictions, impute missing scores or pad failed clips.
No official aggregate is released until technical failures are resolved under the
unchanged protocol. Keep unsuccessful attempts immutable; retries get a new
documented attempt directory and repeat only the failed technical work.

`protocol.json` fixes new locations for the dense feature store, dense teacher
scores, immutable run/evaluation records and canonical summary. All are absent
at freeze. Open final files exclusively and fail on collision; never invoke the
old overwriting extractor, retrain-teacher script or m28 main. Record candidate
manifest, realized keys, per-shard hashes and store identity, raw-file size/metadata
(no bulk video hashing), backbone/adapter/prototype/config/split/index/annotation
hashes, environment, commands, branch/HEAD, dirty state, timings and failures.

The per-video schema in `PER_VIDEO_SCHEMA.json` fixes required record groups.
Retain full-precision attention, teacher scores, logits/probabilities, interval
sets, metrics, coverage, exclusion reasons, and provenance. Model tensors can be
stored separately with immutable path/hash references; summary JSON cannot round
away values needed to recompute metrics. Phase 5 must test metric edge cases and
verify aggregates independently before accepting results. Phase 6 owns final
figures/statistical consolidation; no figure selection by favorable IoU here.
