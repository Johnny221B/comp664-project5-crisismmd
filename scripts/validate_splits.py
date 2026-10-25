#!/usr/bin/env python3
"""Fail-fast validation for canonical ID and California-held-out manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

REQUIRED = {"sample_id", "tweet_id", "image_id", "event", "label", "protocol", "split"}


def validate_frame(frame: pd.DataFrame, source: str = "manifest") -> None:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    if frame.sample_id.astype(str).duplicated().any():
        raise ValueError(f"{source}: duplicate sample_id")
    if set(frame.label.unique()) - {0, 1}:
        raise ValueError(f"{source}: labels must be exactly binary-compatible")
    split_sets = {
        split: part for split, part in frame.groupby("split", sort=False)
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if left not in split_sets or right not in split_sets:
            continue
        for column in ("tweet_id", "image_id"):
            overlap = set(split_sets[left][column].astype(str)) & set(split_sets[right][column].astype(str))
            if overlap:
                raise ValueError(f"{source}: {column} overlap {left}/{right}: {len(overlap)}")
    if set(frame.protocol) == {"ood_heldout"}:
        train_val = frame[frame.split.isin(["train", "val"])]
        if (train_val.event == "california_wildfires").any():
            raise ValueError("California Wildfires leaked into OOD train/val")
        test = frame[frame.split == "test"]
        if len(test) and set(test.event) != {"california_wildfires"}:
            raise ValueError("OOD test contains a non-California event")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    files = sorted(args.manifest_root.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV manifests in {args.manifest_root}")
    for path in files:
        validate_frame(pd.read_csv(path, dtype={"tweet_id": str, "image_id": str}), str(path))
        print(f"PASS {path}")


if __name__ == "__main__":
    main()

