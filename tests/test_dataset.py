"""Contrat : DermDataset(df, data_dir, transform) -> (Tensor, int).

df contient image_id et label (mapping global préparé en amont).
data_dir contient HAM10000_images_part_1/ et HAM10000_images_part_2/.
"""

import random

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.dataset import DermDataset
from src.transforms import get_val_transforms


@pytest.fixture(autouse=True)
def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def samples(tmp_path):
    rows = []
    for label in range(7):
        folder = tmp_path / f"HAM10000_images_part_{label % 2 + 1}"
        folder.mkdir(exist_ok=True)
        image_id = f"image_{label}"
        image = Image.fromarray(
            np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8)
        )
        image.save(folder / f"{image_id}.jpg")
        rows.append({"image_id": image_id, "label": label})
    # Vérifie aussi l'indexation positionnelle après filtrage d'un DataFrame.
    return pd.DataFrame(rows, index=[10, 20, 30, 40, 50, 60, 70]), tmp_path


@pytest.fixture
def dataset(samples):
    df, root = samples
    return DermDataset(df=df, data_dir=str(root), transform=get_val_transforms())


def test_len(dataset):
    assert len(dataset) == 7


@pytest.mark.parametrize("index", range(7))
def test_getitem_retourne_tensor_et_label_correct(dataset, index):
    sample = dataset[index]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    image, label = sample
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert type(label) is int
    assert 0 <= label <= 6
    assert label == index


def test_image_chargee_correspond_a_la_bonne_ligne(samples, dataset):
    _, root = samples
    transform = get_val_transforms()
    for index in range(7):
        path = root / f"HAM10000_images_part_{index % 2 + 1}" / f"image_{index}.jpg"
        with Image.open(path) as image:
            expected = transform(image.convert("RGB"))
        actual, _ = dataset[index]
        torch.testing.assert_close(actual, expected)


def test_image_absente_leve_une_erreur_explicite(samples):
    df, root = samples
    (root / "HAM10000_images_part_1" / "image_0.jpg").unlink()
    # Une validation au constructeur ou au premier accès est acceptée.
    with pytest.raises(FileNotFoundError, match="image_0"):
        dataset = DermDataset(df=df, data_dir=str(root), transform=get_val_transforms())
        dataset[0]


def test_repertoire_absent_leve_une_erreur_explicite(samples):
    df, root = samples
    missing = root / "repertoire_absent"
    with pytest.raises(FileNotFoundError, match="repertoire_absent"):
        dataset = DermDataset(df=df, data_dir=str(missing), transform=get_val_transforms())
        dataset[0]
