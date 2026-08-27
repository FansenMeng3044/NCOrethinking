#!/usr/bin/env python3
"""Reassemble split training-record archives and verify their recorded SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("archives.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    artifact_root = manifest_path.parent.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for archive in manifest["archives"]:
        output = args.output_dir / archive["logical_name"]
        with output.open("wb") as destination:
            for part in archive["parts"]:
                source = artifact_root / part["file"]
                if sha256(source) != part["sha256"]:
                    raise ValueError(f"part hash mismatch: {source}")
                with source.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        destination.write(block)
        actual = sha256(output)
        expected = archive["reconstructed_sha256"]
        if actual != expected:
            raise ValueError(f"archive hash mismatch for {output}: {actual} != {expected}")
        print(f"verified {output} ({output.stat().st_size} bytes, sha256={actual})")


if __name__ == "__main__":
    main()
