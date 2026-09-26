"""Signatures proposées :
get_class_weights(labels, num_classes=7) -> Tensor
save_model(model, path) -> None ; load_model(model, path, device) -> model
compute_metrics(y_true, y_pred) -> dict ; y_pred contient des indices de classes.
Les poids sont proportionnels à 1/effectif ; f1 désigne la macro-F1.
"""

import random

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.utils import compute_metrics, get_class_weights, load_model, save_model


@pytest.fixture(autouse=True)
def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def class_counts():
    # Les sept classes sont présentes, avec des fréquences très différentes.
    return torch.tensor([1, 2, 3, 4, 5, 67, 7])


def test_class_weights_forme_et_valeurs(class_counts):
    labels = torch.repeat_interleave(torch.arange(7), class_counts)
    df = pd.DataFrame({"label": labels.numpy()})
    weights = get_class_weights(df, num_classes=7)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (7,)
    assert weights.is_floating_point()
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()


def test_class_weights_inversement_proportionnels(class_counts):
    labels = torch.repeat_interleave(torch.arange(7), class_counts)
    df = pd.DataFrame({"label": labels.numpy()})
    weights = get_class_weights(df, num_classes=7)
    products = weights.cpu() * class_counts
    # Accepte différentes normalisations globales des poids.
    torch.testing.assert_close(products, products[0].expand_as(products))
    assert weights[0] > weights[5]


@pytest.fixture
def model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 7))


def test_save_model_cree_un_fichier(model, tmp_path):
    path = tmp_path / "model.pt"
    save_model(model, path)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_load_model_restaure_poids_et_predictions(model, tmp_path):
    path = tmp_path / "model.pt"
    model.eval()
    inputs = torch.randn(5, 4)
    with torch.no_grad():
        expected = model(inputs).clone()
    save_model(model, path)

    fresh = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 7))
    assert any(not torch.equal(a, b) for a, b in zip(model.parameters(), fresh.parameters()))
    restored, config = load_model(fresh, path, device=torch.device("cpu"))
    assert isinstance(config, dict)
    assert isinstance(restored, nn.Module)
    assert model.state_dict().keys() == restored.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)


def test_compute_metrics_predictions_parfaites():
    labels = np.arange(7)
    result = compute_metrics(labels, labels.copy())
    assert isinstance(result, dict)
    assert {"accuracy", "balanced_accuracy", "f1"} <= result.keys()
    for key in ("accuracy", "balanced_accuracy", "f1"):
        assert result[key] == pytest.approx(1.0)


def test_compute_metrics_predictions_imparfaites():
    # Classe 0 : 2/3 corrects ; autres classes : rappel parfait.
    truth = np.array([0, 0, 0, 1, 1, 2, 3, 4, 5, 6])
    predicted = np.array([0, 0, 1, 1, 1, 2, 3, 4, 5, 6])
    result = compute_metrics(truth, predicted)
    assert isinstance(result, dict)
    assert {"accuracy", "balanced_accuracy", "f1"} <= result.keys()
    assert result["accuracy"] == pytest.approx(9 / 10)
    assert result["balanced_accuracy"] == pytest.approx((2 / 3 + 1 + 5) / 7)
    assert result["f1"] == pytest.approx((4 / 5 + 4 / 5 + 5) / 7)
