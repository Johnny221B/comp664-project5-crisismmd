from __future__ import annotations

from torchvision.models import ResNet18_Weights


def image_transform(weights_name: str = "IMAGENET1K_V1"):
    weights = getattr(ResNet18_Weights, weights_name)
    return weights.transforms()

