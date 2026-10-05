"""Vérifie data/processed/splits.csv (ignoré si le fichier est absent)."""

from pathlib import Path

import pandas as pd
import pytest

from src.classes import CLASSES

SPLITS_PATH = Path(__file__).resolve().parents[1] / "data/processed/splits.csv"

pytestmark = pytest.mark.skipif(
    not SPLITS_PATH.is_file(), reason="data/processed/splits.csv absent"
)


@pytest.fixture(scope="module")
def splits():
    return pd.read_csv(SPLITS_PATH)


@pytest.fixture(scope="module")
def train_val(splits):
    return splits[splits["split"] == "train_val"]


def test_colonnes(splits):
    assert list(splits.columns) == ["image_id", "lesion_id", "dx", "label", "fold", "split"]


def test_images_uniques(splits):
    assert not splits["image_id"].duplicated().any()


def test_aucune_lesion_commune_entre_train_val_et_test(splits, train_val):
    test = splits[splits["split"] == "test"]
    assert len(test) > 0
    assert set(train_val["lesion_id"]).isdisjoint(test["lesion_id"])


def test_chaque_lesion_dans_un_seul_fold(train_val):
    folds_par_lesion = train_val.groupby("lesion_id")["fold"].nunique()
    assert (folds_par_lesion == 1).all()


def test_les_sept_classes_dans_chaque_fold(train_val):
    for fold in range(5):
        presentes = set(train_val.loc[train_val["fold"] == fold, "dx"])
        assert presentes == set(CLASSES), f"fold {fold} : {set(CLASSES) - presentes}"


def test_label_correspond_a_classes(splits):
    attendu = splits["dx"].map(CLASSES.index)
    assert (splits["label"] == attendu).all()
