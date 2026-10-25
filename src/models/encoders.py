from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel
from torchvision.models import ResNet18_Weights, resnet18


class TextEncoder(nn.Module):
    output_dim = 768

    def __init__(self, name: str = "distilbert-base-uncased"):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.model.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]


class ImageEncoder(nn.Module):
    output_dim = 512

    def __init__(self, weights_name: str = "IMAGENET1K_V1"):
        super().__init__()
        network = resnet18(weights=getattr(ResNet18_Weights, weights_name))
        network.fc = nn.Identity()
        network.requires_grad_(False)
        self.model = network

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(image)

