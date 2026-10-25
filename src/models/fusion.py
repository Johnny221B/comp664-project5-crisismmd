from __future__ import annotations

import torch
from torch import nn
from .encoders import ImageEncoder, TextEncoder


class LateFusionClassifier(nn.Module):
    def __init__(self, text_model: str, image_weights: str, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.text_encoder = TextEncoder(text_model)
        self.image_encoder = ImageEncoder(image_weights)
        self.head = nn.Sequential(nn.Linear(1280, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, image: torch.Tensor, **_) -> torch.Tensor:
        text = self.text_encoder(input_ids, attention_mask)
        visual = self.image_encoder(image)
        return self.head(torch.cat([text, visual], dim=1))
