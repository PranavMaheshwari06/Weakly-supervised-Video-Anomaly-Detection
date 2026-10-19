# Normal-Bag Response-Loss Study — Evidence Mirror (Validation Only)

This directory is the Git-visible, small-file mirror for a post-hoc,
validation-only controlled study, executed after the official protocol was
frozen. It
is **not** part of the official `dispatch_repro_v1` evidence tier and must not
be conflated with it. No checkpoint, feature shard, or teacher-score tensor is
included here; those remain local under `artifacts/stage2/runs/` and
`artifacts/stage2/eval/` (see the main README for data access).

## What this tests

The thesis Discussion section "Response Prediction and the Role of Binary
Gating" diagnosed, but did not test, universal
Police over-prediction as a consequence of the response loss being restricted
to anomalous bags. This study tests that diagnosis directly, in two stages,
entirely on the 161-video validation split. The test split was never loaded:
each run's config sets `reproducibility.validation_only: true`, which
`stage2_class/src/eval/m29_eval_dispatch.py` enforces by raising an error if
`--split test` is requested against it.

## The three arms

- **control** (`ablation_full_s{1337,2027,3407}`): the existing Full arm of
  the teacher-mechanism ablation (Section 5.7 of the thesis), response loss
  restricted to anomalous bags only — architecturally and procedurally
  identical to the official protocol's loss, just not one of the three
  official checkpoints.
- **tier1a_unweighted** (`abl_normal_response_v1_s{1337,2027,3407}`): response
  loss extended to normal bags (all-zero target), as one merged mean over the
  whole batch.
- **tier1c_reweighted** (`abl_normal_response_reweighted_v1_s{1337,2027,3407}`):
  response loss extended to normal bags as two separately averaged terms —
  the anomalous term at its original, undiluted strength, the normal term
  added at a pre-registered weight of 0.3.

All three arms share seeds, architecture, optimizer, early-stopping
configuration, and training-bag-plan identity; only the response-loss
treatment differs. Per-seed numbers are in `summary.json`; per-run
configuration, checkpoint, and evaluation hashes are in
`reproducibility/PROVENANCE.md` (Section 8).

## Result in one line

Tier 1a removes raw universal Police activation but shows no improvement over
control once matched for what binary gating already provides; Tier 1c tested
whether loss dilution explained that null result and found it did not — the
reweighted variant traded a small, statistically marginal gain for a real
regression in Police and Medical precision. Both are reported as validation-only,
non-official findings.

## Files

- `summary.json`: per-arm, per-seed metrics and 3-seed aggregate mean/sample_std,
  built by `utils/build_normal_bag_evidence.py` directly from the (locally retained, gitignored) `dispatch_eval_val.json`
  outputs of `m29_eval_dispatch.py --split val`.

Whole-file SHA-256:

| File | SHA-256 |
| --- | --- |
| `summary.json` | `8123b0bec4e1da314b013fcb4e423e871b625231994501f2fd4ca3f71e9b6865` |

Rebuild and re-verify from the repository root:

```bash
PYTHONPATH=. venv/bin/python utils/build_normal_bag_evidence.py
```
