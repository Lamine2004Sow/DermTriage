"""Signatures proposées :
train_one_epoch(model, loader, criterion, optimizer, device) -> float
evaluate(model, loader, criterion, device) -> dict
La loss est moyennée par exemple ; f1 est la macro-F1.
"""

import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import evaluate, train_one_epoch


@pytest.fixture(autouse=True)
def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def setup():
    images = torch.randn(11, 3, 8, 8)
    labels = torch.arange(11) % 7
    # Dernier lot incomplet pour vérifier la moyenne par exemple.
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 7))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, loader, criterion, optimizer, torch.device("cpu")


def test_train_retourne_loss_float_finie(setup):
    model, loader, criterion, optimizer, device = setup
    loss = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    assert type(loss) is float
    assert np.isfinite(loss)
    assert loss >= 0


def test_train_met_a_jour_les_parametres(setup):
    model, loader, criterion, optimizer, device = setup
    before = [p.detach().clone() for p in model.parameters()]
    train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_evaluate_retourne_les_metriques_sans_modifier_les_poids(setup):
    model, loader, criterion, _, device = setup
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = evaluate(model, loader, criterion, device)
    assert isinstance(result, dict)
    assert {"loss", "accuracy", "balanced_accuracy", "f1"} <= result.keys()
    assert np.isfinite(result["loss"])
    assert result["loss"] >= 0
    for key in ("accuracy", "balanced_accuracy", "f1"):
        assert 0 <= result[key] <= 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["train", "eval"])
def test_loss_moyenne_par_exemple(setup, phase):
    model, loader, criterion, _, device = setup
    images, labels = loader.dataset.tensors
    with torch.no_grad():
        expected = criterion(model(images), labels).item()
    if phase == "train":
        # lr=0 garde les logits fixes pour une référence calculable exactement.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        actual = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
    else:
        actual = evaluate(model, loader, criterion, device)["loss"]
    assert actual == pytest.approx(expected, rel=1e-6)
