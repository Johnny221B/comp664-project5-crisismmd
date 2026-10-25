#!/usr/bin/env python3
"""Audit canonical manifests, image readability, and split leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def overlaps(frame: pd.DataFrame, column: str) -> dict[str, int]:
    groups = {name: set(part[column].astype(str)) for name, part in frame.groupby("split")}
    return {
        f"{a}_{b}": len(groups.get(a, set()) & groups.get(b, set()))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stats-out", type=Path, default=Path("results/data_stats.json"))
    args = parser.parse_args()
    manifests = sorted(args.manifest_root.glob("*.csv"))
    if not manifests:
        raise SystemExit("no manifests found; run prepare_manifests.py first")
    all_stats: dict[str, object] = {}
    exceptions = []
    image_meta = Counter()
    checksums: dict[str, str] = {}
    id_frame = pd.read_csv(args.manifest_root / "id_official.csv", dtype={"tweet_id": str, "image_id": str})
    for row in id_frame.itertuples():
        path = Path(row.image_path)
        try:
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
                image_meta[f"format:{image.format or 'unknown'}"] += 1
                image_meta[f"size:{image.width}x{image.height}"] += 1
            checksums[row.sample_id] = file_sha256(path)
        except Exception as error:
            exceptions.append({"sample_id": row.sample_id, "image_path": str(path), "error": repr(error)})
    for manifest in manifests:
        frame = pd.read_csv(manifest, dtype={"tweet_id": str, "image_id": str})
        stats = {
            "rows": len(frame),
            "unique_tweets": int(frame.tweet_id.nunique()),
            "events": sorted(frame.event.unique().tolist()),
            "event_split_label_counts": [
                {"event": e, "split": s, "label": int(label), "n": int(n)}
                for (e, s, label), n in frame.groupby(["event", "split", "label"]).size().items()
            ],
            "tweet_overlap": overlaps(frame, "tweet_id"),
            "image_id_overlap": overlaps(frame, "image_id"),
            "text_length_percentiles": frame.text.str.len().quantile([0, .25, .5, .75, .9, .95, .99, 1]).to_dict(),
        }
        if checksums:
            mapped = frame.assign(image_checksum=frame.sample_id.map(checksums))
            stats["image_checksum_overlap"] = overlaps(mapped, "image_checksum")
        all_stats[manifest.stem] = stats
    all_stats["images"] = {
        "checked": len(id_frame), "readable": len(id_frame) - len(exceptions),
        "exceptions": len(exceptions), "format_counts": dict(sorted((k[7:], v) for k, v in image_meta.items() if k.startswith("format:"))),
        "distinct_shapes": sum(k.startswith("size:") for k in image_meta),
    }
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(json.dumps(all_stats, indent=2, sort_keys=True) + "\n")
    exception_path = args.stats_out.parent / "image_exceptions.csv"
    pd.DataFrame(exceptions, columns=["sample_id", "image_path", "error"]).to_csv(exception_path, index=False)
    download_manifest = args.data_root / "download_manifest.json"
    download = json.loads(download_manifest.read_text()) if download_manifest.exists() else {"status": "download manifest missing"}
    lines = [
        "# CrisisMMD v2.0 Data Audit", "", "## Provenance", "",
        f"- Download manifest: `{download_manifest}`", f"- Source record: `{json.dumps(download)}`", "",
        "## Canonical data", "",
    ]
    for protocol, stats in all_stats.items():
        if protocol == "images":
            continue
        lines += [f"### {protocol}", "", f"- Rows: {stats['rows']}", f"- Unique tweets: {stats['unique_tweets']}",
                  f"- Events: {', '.join(stats['events'])}", f"- Tweet overlap: `{stats['tweet_overlap']}`",
                  f"- Image ID overlap: `{stats['image_id_overlap']}`", f"- Image checksum overlap: `{stats.get('image_checksum_overlap')}`", ""]
    lines += ["## Image validation", "", f"- Checked: {len(id_frame)}", f"- Readable: {len(id_frame)-len(exceptions)}",
              f"- Exceptions: {len(exceptions)} (see `{exception_path}`)", f"- Formats: `{all_stats['images']['format_counts']}`", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"WROTE {args.out}, {args.stats_out}, {exception_path}")


if __name__ == "__main__":
    main()
