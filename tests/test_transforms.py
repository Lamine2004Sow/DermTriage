"""Contrat : get_train_transforms() et get_val_transforms() -> callable."""

import random

import numpy as np
import pytest
import torch
from PIL import Image

from src.transforms import get_train_transforms, get_val_transforms


@pytest.fixture(autouse=True)
def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def image():
    return Image.fromarray(
        np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8)
    )


@pytest.mark.parametrize("factory", [get_train_transforms, get_val_transforms])
def test_transformation_callable(factory):
    assert callable(factory())


@pytest.mark.parametrize("factory", [get_train_transforms, get_val_transforms])
def test_forme_type_et_intervalle(image, factory):
    result = factory()(image)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    assert result.min().item() >= -3
    assert result.max().item() <= 3


@pytest.mark.parametrize("value", [0, 255])
def test_validation_utilise_la_normalisation_imagenet(value):
    # L'intervalle [-3, 3] seul accepterait aussi des pixels non normalisés.
    image = Image.fromarray(np.full((450, 600, 3), value, dtype=np.uint8))
    actual = get_val_transforms()(image)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    expected = ((value / 255 - mean) / std)[:, None, None].expand(3, 224, 224)
    torch.testing.assert_close(actual, expected)


def test_validation_deterministe(image):
    transform = get_val_transforms()
    expected = transform(image)
    for _ in range(3):
        torch.testing.assert_close(transform(image), expected, rtol=0, atol=0)


@pytest.mark.parametrize("factory", [get_train_transforms, get_val_transforms])
def test_image_source_inchangee(image, factory):
    before = np.asarray(image).copy()
    factory()(image)
    np.testing.assert_array_equal(np.asarray(image), before)
