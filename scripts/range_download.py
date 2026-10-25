#!/usr/bin/env python3
"""Deterministic parallel HTTP range downloader for large official artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import urllib.request
from pathlib import Path


def fetch(url: str, part: Path, start: int, end: int) -> None:
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "COMP664-CrisisMMD/1.0"})
    temporary = part.with_suffix(".tmp")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        if response.status != 206:
            raise RuntimeError(f"server ignored Range request for {start}-{end}: {response.status}")
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if temporary.stat().st_size != expected:
        raise IOError(f"short part {part}: {temporary.stat().st_size} != {expected}")
    temporary.replace(part)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    part_root = args.output.with_suffix(args.output.suffix + ".parts")
    part_root.mkdir(parents=True, exist_ok=True)
    chunk_size = (args.bytes + args.workers - 1) // args.workers
    jobs = []
    for index in range(args.workers):
        start = index * chunk_size
        end = min(args.bytes - 1, start + chunk_size - 1)
        if start <= end:
            jobs.append((args.url, part_root / f"part-{index:03d}", start, end))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, *job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    temporary = args.output.with_suffix(args.output.suffix + ".assembling")
    with temporary.open("wb") as output:
        for _, part, _, _ in jobs:
            with part.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    output.write(chunk)
    if temporary.stat().st_size != args.bytes:
        raise IOError("assembled file has unexpected size")
    temporary.replace(args.output)
    print(f"WROTE {args.output} bytes={args.output.stat().st_size}")


if __name__ == "__main__":
    main()
