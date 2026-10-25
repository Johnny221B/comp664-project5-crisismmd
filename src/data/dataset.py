from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class PairedCrisisDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, transform: Any, max_length: int = 128):
        self.frame = frame.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        text = str(row.text).strip()
        if not text:
            raise ValueError(f"empty text for {row.sample_id}")
        encoded = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        with Image.open(Path(row.image_path)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixels = self.transform(image)
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "image": pixels,
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "sample_id": str(row.sample_id), "tweet_id": str(row.tweet_id),
            "event": str(row.event), "split": str(row.split),
        }


def read_manifest(path: str | Path, split: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str, "tweet_id": str, "image_id": str})
    required = {"sample_id", "tweet_id", "event", "text", "image_path", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if split is not None:
        frame = frame[frame.split == split].copy()
    if set(frame.label.unique()) - {0, 1}:
        raise ValueError("non-binary label in manifest")
    return frame.reset_index(drop=True)

