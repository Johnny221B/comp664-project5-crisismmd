from __future__ import annotations

import torch
from torch import nn


class PresenceAwareFusionHead(nn.Module):
    def __init__(self, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1282, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, features: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([features, presence], dim=1))


def apply_modality_dropout(features: torch.Tensor, probability: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    result = features.clone(); batch = len(result)
    presence = torch.ones(batch, 2, dtype=result.dtype, device=result.device)
    drop = torch.rand(batch, generator=generator, device="cpu") < probability
    modality = torch.randint(0, 2, (batch,), generator=generator, device="cpu")
    text_drop = (drop & modality.eq(0)).to(result.device); image_drop = (drop & modality.eq(1)).to(result.device)
    result[text_drop, :768] = 0; presence[text_drop, 0] = 0
    result[image_drop, 768:] = 0; presence[image_drop, 1] = 0
    return result, presence


def condition_features(features: torch.Tensor, condition: str) -> tuple[torch.Tensor, torch.Tensor]:
    result = features.clone(); presence = torch.ones(len(result), 2, dtype=result.dtype, device=result.device)
    if condition == "text_only": result[:, 768:] = 0; presence[:, 1] = 0
    elif condition == "image_only": result[:, :768] = 0; presence[:, 0] = 0
    elif condition != "both": raise ValueError(condition)
    return result, presence

