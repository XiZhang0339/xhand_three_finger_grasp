#!/usr/bin/env python3
"""Verify the portable grasp-pose bundle using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Optionally verify xhand_left.xml and uv.lock in an xhand1 checkout.",
    )
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parent
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []

    for relative, metadata in sorted(manifest["files"].items()):
        path = (bundle_root / relative).resolve()
        if bundle_root not in path.parents:
            failures.append(f"unsafe manifest path: {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256_file(path)
        expected = metadata["sha256"]
        if actual != expected:
            failures.append(f"hash mismatch: {relative}: {actual} != {expected}")

    if args.repo_root is not None:
        repo_root = args.repo_root.resolve()
        for relative, metadata in sorted(manifest["runtime_bindings"].items()):
            path = repo_root / relative
            if not path.is_file():
                failures.append(f"missing runtime binding: {path}")
                continue
            actual = sha256_file(path)
            expected = metadata["sha256"]
            if actual != expected:
                failures.append(
                    f"runtime hash mismatch: {relative}: {actual} != {expected}"
                )

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1

    print(f"OK {manifest['bundle_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
