#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.prepare import build_ood, load_official_agreed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    identity = load_official_agreed(args.data_root.resolve())
    ood = build_ood(identity, args.seed)
    for name, frame in (("id_official.csv", identity), ("ood_heldout.csv", ood)):
        target = args.out / name
        frame.to_csv(target, index=False)
        print(f"WROTE {target} rows={len(frame)}")


if __name__ == "__main__":
    main()

