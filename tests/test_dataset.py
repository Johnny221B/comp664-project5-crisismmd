import pandas as pd
import torch
from PIL import Image

from src.data.dataset import PairedCrisisDataset


class Tokenizer:
    def __call__(self, text, **kwargs):
        assert text
        length = kwargs["max_length"]
        return {"input_ids": torch.ones(1, length, dtype=torch.long), "attention_mask": torch.ones(1, length, dtype=torch.long)}


def test_paired_item(tmp_path):
    path = tmp_path / "image.jpg"
    Image.new("RGB", (16, 16), "red").save(path)
    frame = pd.DataFrame([{"sample_id": "s", "tweet_id": "t", "event": "e", "text": "hello", "image_path": path, "label": 1, "split": "train"}])
    dataset = PairedCrisisDataset(frame, Tokenizer(), lambda image: torch.zeros(3, 8, 8), max_length=4)
    item = dataset[0]
    assert item["input_ids"].shape == (4,)
    assert item["image"].shape == (3, 8, 8)
    assert item["label"].item() == 1

