#!/usr/bin/env python3
"""Strict content-checksum leakage gate for canonical manifests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def checksum(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_content(frame: pd.DataFrame, source: str) -> None:
    if "image_checksum" not in frame:
        frame = frame.copy()
        frame["image_checksum"] = [checksum(path) for path in frame.image_path]
    splits = {name: set(part.image_checksum) for name, part in frame.groupby("split")}
    failures = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = splits.get(left, set()) & splits.get(right, set())
        if overlap:
            failures.append(f"{left}/{right}={len(overlap)}")
    if failures:
        raise ValueError(f"{source}: cross-split image content leakage: {', '.join(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    for path in sorted(args.manifest_root.glob("*.csv")):
        try:
            validate_content(pd.read_csv(path), str(path))
            print(f"PASS {path}")
        except ValueError as error:
            print(f"FAIL {error}")
            failures.append(error)
    if failures:
        raise SystemExit(f"{len(failures)} manifest(s) failed content leakage validation")


if __name__ == "__main__":
    main()

