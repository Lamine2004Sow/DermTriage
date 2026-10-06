"""scripts/extract_features.py (E1) sur un mini jeu de données factice, sans poids ImageNet."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.extract_features import build_encoder, main
from src.classes import CLASSES


@pytest.fixture
def mini(tmp_path):
    rng = np.random.default_rng(0)
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    rows = []
    for i in range(28):
        image_id = f"img_{i}"
        Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(img_dir / f"{image_id}.jpg")
        rows.append({"image_id": image_id, "lesion_id": f"les_{i}", "dx": CLASSES[i % 7],
                     "label": i % 7, "fold": i % 2, "split": "train_val"})
    csv = tmp_path / "train_val.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, img_dir, tmp_path


def launch(mini, *extra):
    csv, img_dir, tmp = mini
    return main(["--train-val", str(csv), "--image-dir", str(img_dir), "--cache", str(tmp / "f.npz"),
                 "--out-dir", str(tmp / "out"), "--num-workers", "0", "--batch-size", "5",
                 "--no-pretrained", *extra])


def test_encodeur_sort_512_dimensions():
    import torch
    encoder = build_encoder(pretrained=False)
    assert encoder(torch.zeros(2, 3, 224, 224)).shape == (2, 512)
    assert not encoder.training


def test_ecrit_cache_et_metriques_par_fold(mini):
    results = launch(mini)
    tmp = mini[2]
    saved = np.load(tmp / "f.npz")
    assert saved["features"].shape == (28, 512)
    assert [r["fold"] for r in results] == [0, 1]
    for fold in (0, 1):
        metrics = json.loads((tmp / "out" / f"fold{fold}_seed0" / "metrics.json").read_text())
        assert {"fold", "n_train", "n_val", "accuracy", "balanced_accuracy", "f1"} <= metrics.keys()
        assert metrics["n_val"] == 14
        logits = pd.read_csv(tmp / "out" / f"fold{fold}_seed0" / "val_logits.csv")
        assert list(logits.columns) == ["image_id", "lesion_id", "label"] + [f"logit_{i}" for i in range(7)]
        assert len(logits) == 14


def test_deuxieme_lancement_relit_le_cache(mini, capsys):
    launch(mini)
    capsys.readouterr()
    launch(mini)
    assert "lues dans" in capsys.readouterr().out


def test_refuse_une_ligne_de_test(mini):
    csv = mini[0]
    df = pd.read_csv(csv)
    df.loc[0, "split"] = "test"
    df.to_csv(csv, index=False)
    with pytest.raises(AssertionError):
        launch(mini)
