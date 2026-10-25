#!/usr/bin/env python3
"""Create explicitly renamed content-disjoint sensitivity manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def checksum(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_strict(frame: pd.DataFrame, protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["image_checksum"] = [checksum(path) for path in frame.image_path]
    priority = {"train": 0, "val": 1, "test": 2}
    keep_split = frame.groupby("image_checksum").split.transform(
        lambda values: max(values, key=priority.__getitem__)
    )
    removed = frame.loc[frame.split.ne(keep_split)].copy()
    strict = frame.loc[frame.split.eq(keep_split)].copy()
    strict["protocol"] = protocol
    return strict.reset_index(drop=True), removed.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    summary = {}
    for source_name, output_name, protocol in (
        ("id_official.csv", "id_official_strict.csv", "id_official_strict"),
        ("ood_heldout.csv", "ood_heldout_strict.csv", "ood_heldout_strict"),
    ):
        source = pd.read_csv(args.manifest_root / source_name, dtype={"tweet_id": str, "image_id": str})
        strict, removed = make_strict(source, protocol)
        strict.to_csv(args.manifest_root / output_name, index=False)
        args.results_root.mkdir(parents=True, exist_ok=True)
        removed.to_csv(args.results_root / output_name.replace(".csv", "_removed.csv"), index=False)
        summary[protocol] = {
            "input_rows": len(source), "output_rows": len(strict), "removed_rows": len(removed),
            "removed_by_split": removed.groupby("split").size().to_dict(),
        }
        print(protocol, summary[protocol])
    (args.results_root / "strict_manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

