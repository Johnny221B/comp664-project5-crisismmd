#!/usr/bin/env python3
"""Download CrisisMMD artifacts without overwriting existing files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

OFFICIAL = {
    "CrisisMMD_v2.0.tar.gz": "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz",
    "crisismmd_datasplit_agreed_label.zip": "https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_agreed_label.zip",
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "COMP664-CrisisMMD/1.0"})
    mode = "ab" if partial.exists() else "wb"
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response, partial.open(mode) as output:
        if offset and response.status != 206:
            output.close()
            partial.unlink()
            return download(url, destination)
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    partial.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                if not (base == (base / member.name).resolve() or base in (base / member.name).resolve().parents):
                    raise ValueError(f"unsafe archive member: {member.name}")
            bundle.extractall(destination, filter="data")
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if not (base == (base / name).resolve() or base in (base / name).resolve().parents):
                    raise ValueError(f"unsafe archive member: {name}")
            bundle.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("official", "hf", "local"), default="official")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()
    if args.source != "official":
        raise NotImplementedError("M0 currently accepts only the verified official source")
    root = args.data_root.resolve()
    archives, splits = root / "archives", root / "splits"
    archives.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, url in OFFICIAL.items():
        target = (splits if filename.endswith(".zip") else archives) / filename
        download(url, target)
        records.append({
            "filename": filename, "path": str(target), "url": url,
            "bytes": target.stat().st_size, "sha256": sha256(target),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        })
        if not args.no_extract:
            safe_extract(target, splits / "agreed_label" if target.suffix == ".zip" else root / "extracted")
    manifest = root / "download_manifest.json"
    manifest.write_text(json.dumps({"source": "official", "artifacts": records}, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "artifacts": records}, indent=2))


if __name__ == "__main__":
    main()

