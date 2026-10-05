"""Contrat : build_model(name, pretrained, trainable) -> nn.Module à 7 sorties."""

import pytest
import torch

from src.models import build_model


def trainable_blocks(model):
    return {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}


def test_sortie_a_sept_classes():
    model = build_model(pretrained=False, trainable="head").eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 7)


def test_head_ne_laisse_que_fc_entrainable():
    assert trainable_blocks(build_model(pretrained=False, trainable="head")) == {"fc"}


def test_layer4_entraine_layer4_et_fc():
    model = build_model(pretrained=False, trainable="layer4")
    assert trainable_blocks(model) == {"layer4", "fc"}


def test_all_entraine_tout():
    model = build_model(pretrained=False, trainable="all")
    assert all(p.requires_grad for p in model.parameters())


def test_arguments_invalides():
    with pytest.raises(ValueError):
        build_model(pretrained=False, trainable="tout")
    with pytest.raises(ValueError):
        build_model(name="vgg", pretrained=False)
