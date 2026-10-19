from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml


RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_run_tag(run_tag: str) -> str:
    value = run_tag.strip()
    if not value or not RUN_TAG_PATTERN.fullmatch(value):
        raise ValueError(
            "run_tag must start with a letter or number and contain only letters, "
            "numbers, '.', '_' or '-'"
        )
    return value


def require_empty_run_directory(run_root: str | Path) -> Path:
    path = Path(run_root)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            f"Refusing to reuse non-empty run directory: {path}. Choose a unique run tag."
        )
    return path


def feature_store_identity(store_root: str | Path, split: str) -> Dict:
    root = Path(store_root)
    shard_paths = sorted(root.glob(f"global_{split}_shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No feature shards found for split={split} under {root}")

    files: List[Dict[str, str]] = []
    for path in shard_paths:
        files.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        })

    return {
        "root": str(root),
        "split": split,
        "files": files,
        "identity_sha256": sha256_json(files),
    }


def hash_files(paths: Iterable[str | Path]) -> Dict[str, str]:
    return {str(Path(path)): sha256_file(path) for path in paths}


def git_state() -> Dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def software_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "executable": sys.executable,
    }
