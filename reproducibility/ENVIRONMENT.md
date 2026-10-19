# Reproducibility Environment Record

Captured: 2026-09-05 on the project host.

This record separates the environment embedded in the official run manifests from the live environment observed on 2026-09-05. Versions not recorded by the run manifests must not be retroactively claimed as exact training-time versions.

## Official run-manifest record

All three `dispatch_repro_*` manifests agree on:

| Item | Recorded value |
| --- | --- |
| Python | `3.12.3` |
| Executable | `/home/pranavmaheshwari/Pranav_master_thesis/master_thesis_crowd_anomaly/venv/bin/python` |
| Platform | `Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39` |
| PyTorch | `2.5.1+cu121` |
| PyTorch CUDA build | CUDA 12.1, encoded by the Torch build suffix |
| NumPy | `1.26.4` |
| PyYAML | `6.0.2` |
| GPU name | `NVIDIA GeForce RTX 4050 Laptop GPU` |
| Implementation commit | `75ef4dfc1c839b1b1a100cc291904be5c9e1dec4` |
| Implementation tree | `git.dirty: true` |

The manifests did not record the NVIDIA driver, dedicated VRAM, cuDNN, Torchvision, Transformers, PEFT, or several other library versions.

## Live host (2026-09-05)

| Item | Observed value |
| --- | --- |
| OS | Ubuntu 24.04.3 LTS under WSL2 |
| Kernel | `5.15.167.4-microsoft-standard-WSL2` |
| Architecture | `x86_64` |
| Python | `3.12.3` |
| PyTorch | `2.5.1+cu121` |
| PyTorch CUDA runtime | `12.1` |
| cuDNN | `90100` (cuDNN 9.1.0) |
| CUDA available outside the sandbox | `true`; one device |
| GPU | `NVIDIA GeForce RTX 4050 Laptop GPU` |
| Dedicated GPU memory | `6141 MiB` reported by `nvidia-smi`; PyTorch reports `6,438,780,928` bytes (`6140.5 MiB`) |
| NVIDIA driver | `591.66` |
| `nvidia-smi` client / driver-advertised CUDA | `590.51` / `13.1` |

The dedicated capacity is approximately 6.0 GiB. No shared system memory is included in that value or represented as CUDA VRAM. The ordinary sandbox blocked NVML and reported CUDA unavailable; the GPU values above came from read-only host-level `nvidia-smi` and PyTorch probes.

Relevant installed packages:

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
numpy==1.26.4
pandas==2.2.2
PyYAML==6.0.2
opencv-python==4.10.0.84
Pillow==10.4.0
transformers==5.12.0
accelerate==1.14.0
einops==0.8.2
scikit-learn==1.8.0
matplotlib==3.10.8
peft==0.19.1
scipy==1.17.0
rich==13.8.0
tqdm==4.66.5
```

`venv/bin/python -m pip check` reports no broken requirements.

## Requirements file

The original `requirements.txt` (SHA-256 `d6668f61b80a38595384edb3e5673be3e7f7ac1b6f385a44fbc8465397a29da9`) pinned `torch==2.3.1` and `torchvision==0.18.1`, which conflicted with the run manifests and the live environment, and left several packages unpinned. The submitted `requirements.txt` pins the live versions listed above. Only the versions in the "Official run-manifest record" section are proven training-time versions; the other pins describe the environment in which the evidence was verified and the dense temporal work was executed.
