"""Lancement de bout en bout de scripts/train.py sur un mini jeu de données factice."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from scripts.train import main
from src.classes import CLASSES


@pytest.fixture
def mini(tmp_path):
    rng = np.random.default_rng(0)
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    rows = []
    for i in range(20):
        image_id = f"img_{i}"
        Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(img_dir / f"{image_id}.jpg")
        rows.append({"image_id": image_id, "lesion_id": f"les_{i}", "dx": CLASSES[i % 7],
                     "label": i % 7, "fold": i % 2, "split": "train_val"})
    csv = tmp_path / "train_val.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    cfg = {"name": "mini", "model": {"arch": "resnet18", "pretrained": False, "trainable": "head"},
           "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
           "train": {"epochs": 2, "batch_size": 4, "patience": 2, "num_workers": 0},
           "loss": {"class_weights": False}, "augment": False, "image_dir": str(img_dir)}
    cfg_path = tmp_path / "mini.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path, csv, tmp_path / "runs"


def launch(mini, seed=0):
    cfg_path, csv, runs = mini
    return main(["--config", str(cfg_path), "--fold", "0", "--seed", str(seed),
                 "--runs-dir", str(runs), "--train-val", str(csv)])


def test_ecrit_les_cinq_fichiers(mini):
    out = launch(mini)
    assert out.name == "fold0_seed0" and out.parent.name == "mini"
    for name in ("config.yaml", "history.csv", "best.pt", "val_logits.csv", "metrics.json"):
        assert (out / name).is_file(), name

    logits = pd.read_csv(out / "val_logits.csv")
    assert list(logits.columns) == ["image_id", "lesion_id", "label"] + [f"logit_{i}" for i in range(7)]
    assert len(logits) == 10
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["fold"] == 0 and "git_commit" in cfg
    metrics = json.loads((out / "metrics.json").read_text())
    assert 0 <= metrics["f1"] <= 1


def test_deux_lancements_identiques_memes_chiffres(mini, tmp_path):
    first = json.loads((launch(mini) / "metrics.json").read_text())
    cfg_path, csv, _ = mini
    again = main(["--config", str(cfg_path), "--fold", "0", "--seed", "0",
                  "--runs-dir", str(tmp_path / "runs2"), "--train-val", str(csv)])
    second = json.loads((again / "metrics.json").read_text())
    assert first["f1"] == pytest.approx(second["f1"], abs=1e-4)
    assert first["loss"] == pytest.approx(second["loss"], abs=1e-4)


def test_le_test_n_est_jamais_charge(mini):
    cfg_path, csv, runs = mini
    df = pd.read_csv(csv)
    df.loc[0, "split"] = "test"
    df.to_csv(csv, index=False)
    with pytest.raises(AssertionError):
        launch(mini)
