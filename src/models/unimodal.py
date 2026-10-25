from __future__ import annotations

import torch
from torch import nn
from .encoders import ImageEncoder, TextEncoder


class TextClassifier(nn.Module):
    def __init__(self, text_model: str, dropout: float = 0.2):
        super().__init__()
        self.encoder = TextEncoder(text_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 2))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_) -> torch.Tensor:
        return self.head(self.encoder(input_ids, attention_mask))


class ImageClassifier(nn.Module):
    def __init__(self, image_weights: str, dropout: float = 0.2):
        super().__init__()
        self.encoder = ImageEncoder(image_weights)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 2))

    def forward(self, image: torch.Tensor, **_) -> torch.Tensor:
        return self.head(self.encoder(image))

