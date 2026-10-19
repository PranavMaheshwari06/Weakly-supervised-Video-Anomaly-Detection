from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPOSITORY_ROOT / "reproducibility/evidence/ARTIFACT_SHA256_INVENTORY.json"
# The inventory predates the move of the evidence mirror to reproducibility/evidence.
MOVED = ("RP_docs/evidence/", "reproducibility/evidence/")
# Inventoried files that are intentionally not part of the submitted repository.
SUPERSEDED = {
    "requirements.txt": "replaced by versions pinned to the recorded environment (reproducibility/ENVIRONMENT.md)",
    "RP_docs/09_dispatch_reproducibility_evidence.md": "working report; its evidence is mirrored in reproducibility/evidence/dispatch_repro_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_store_identity(root: Path, split: str) -> tuple[int, str]:
    files = []
    for path in sorted(root.glob(f"global_{split}_shard_*.pt")):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(files), hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the selective thesis evidence inventory.")
    parser.add_argument("inventory", nargs="?", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--skip-feature-stores",
        action="store_true",
        help="Verify direct file hashes only; skip feature-shard identity recomputation.",
    )
    args = parser.parse_args()

    inventory_path = args.inventory.resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    verified_files = 0

    for entry in inventory["files"]:
        if entry["path"] in SUPERSEDED:
            print(f"SKIP (superseded): {entry['path']}: {SUPERSEDED[entry['path']]}")
            continue
        path = REPOSITORY_ROOT / entry["path"].replace(*MOVED)
        if not path.is_file():
            failures.append(f"missing file: {entry['path']}")
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            failures.append(
                f"hash mismatch: {entry['path']} expected={entry['sha256']} actual={actual}"
            )
            continue
        verified_files += 1

    verified_stores = 0
    if not args.skip_feature_stores:
        for entry in inventory["feature_stores"]:
            root = REPOSITORY_ROOT / entry["root"]
            count, identity = feature_store_identity(root, entry["split"])
            if count != entry["file_count"]:
                failures.append(
                    f"feature count mismatch: {entry['root']} split={entry['split']} "
                    f"expected={entry['file_count']} actual={count}"
                )
                continue
            if identity != entry["identity_sha256"]:
                failures.append(
                    f"feature identity mismatch: {entry['root']} split={entry['split']} "
                    f"expected={entry['identity_sha256']} actual={identity}"
                )
                continue
            verified_stores += 1

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    feature_message = "skipped" if args.skip_feature_stores else str(verified_stores)
    print(f"PASS: verified {verified_files} files; feature-store identities={feature_message}")


if __name__ == "__main__":
    main()
