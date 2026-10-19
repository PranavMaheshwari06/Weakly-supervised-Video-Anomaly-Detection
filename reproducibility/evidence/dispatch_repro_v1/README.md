# Portable Dispatch Reproducibility Evidence

This directory is the Git-visible, small-file mirror for the official three-seed raw Stage-2 fixed-16-bag protocol. It does not contain checkpoints, feature shards, teacher-score tensors, raw videos, or the clip index.

The official runs are `dispatch_repro_s1337`, `dispatch_repro_s2027`, and `dispatch_repro_s3407`. Their training seeds are 1337, 2027, and 3407; every evaluation uses seed 1337. The selected epochs are 34, 0, and 25. The evidence is official only for raw fixed-bag binary and project-derived Police/Medical/Fire evaluation at threshold 0.5, conditional on the frozen LoRA feature store.

Portable files:

- `summary.json`: byte-identical to `artifacts/stage2/evidence/dispatch_repro_v1/summary.json`.
- `bag_manifests/dispatch_lora_fixed_val.json`: byte-identical to the ignored fixed validation manifest.
- `bag_manifests/dispatch_lora_fixed_test.json`: byte-identical to the ignored fixed test manifest.

Whole-file SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `summary.json` | `ffb1e5d12ba1922a0e21e01eacab83ecd031a9ecc309e26df8a2ea0bea512b5c` |
| `bag_manifests/dispatch_lora_fixed_val.json` | `a4c9174b3491ae302ecec9853c9d7ff526a0ded9b0aa98456bcf2ab409f2e67a` |
| `bag_manifests/dispatch_lora_fixed_test.json` | `9cddad6d40e9efe8f11c51982199ea1a2056a1c5be6131b4d4df083a37fc4041` |

The embedded logical-payload checksums in the manifests are different from these whole-file hashes. Do not substitute one for the other.

Verify the complete selective inventory from the repository root:

```bash
venv/bin/python utils/verify_thesis_evidence.py
```

The large artifacts these files describe (checkpoints, evaluations, feature stores) are not in Git; see the main README for data access.
