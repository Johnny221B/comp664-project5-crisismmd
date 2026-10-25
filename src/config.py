from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    config = yaml.safe_load(path.read_text()) or {}
    base = config.pop("base", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute() and not base_path.exists():
            base_path = path.parent / base_path.name
        merged = load_config(base_path)
        merged.update(config)
        config = merged
    return config

