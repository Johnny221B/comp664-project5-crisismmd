import torch
from src.models.modality_dropout import apply_modality_dropout, condition_features


def test_dropout_never_masks_both_modalities():
    features=torch.ones(100,1280); masked,presence=apply_modality_dropout(features,1.0,torch.Generator().manual_seed(42))
    assert torch.all(presence.sum(1)==1)
    assert torch.all((masked[:,:768].sum(1)==0) ^ (masked[:,768:].sum(1)==0))


def test_missing_conditions_have_explicit_indicators():
    features=torch.ones(2,1280); text,p=condition_features(features,"text_only"); assert torch.all(text[:,768:]==0) and torch.all(p==torch.tensor([1.,0.]))
