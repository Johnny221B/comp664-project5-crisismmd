import torch
from torch import nn

import src.models.fusion as fusion_module
import src.models.unimodal as unimodal_module


class FakeText(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(); self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
    def forward(self, input_ids, attention_mask): return torch.zeros(input_ids.shape[0], 768)


class FakeImage(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(); self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
    def forward(self, image): return torch.zeros(image.shape[0], 512)


def test_three_forward_shapes_and_frozen_encoders(monkeypatch):
    monkeypatch.setattr(unimodal_module, "TextEncoder", FakeText)
    monkeypatch.setattr(unimodal_module, "ImageEncoder", FakeImage)
    monkeypatch.setattr(fusion_module, "TextEncoder", FakeText)
    monkeypatch.setattr(fusion_module, "ImageEncoder", FakeImage)
    inputs = {"input_ids": torch.ones(3, 4, dtype=torch.long), "attention_mask": torch.ones(3, 4, dtype=torch.long), "image": torch.zeros(3, 3, 8, 8)}
    models = [
        unimodal_module.TextClassifier("fake"),
        unimodal_module.ImageClassifier("fake"),
        fusion_module.LateFusionClassifier("fake", "fake"),
    ]
    for model in models:
        logits = model(**inputs)
        assert logits.shape == (3, 2)
        assert torch.isfinite(nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 0])))
        assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if "head" not in name)
