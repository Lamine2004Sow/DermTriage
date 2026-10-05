"""Baselines (DummyClassifier et régression logistique sur features couleur).

Lit data/processed/train_val.csv (le test n'est jamais chargé), entraîne sur les folds 1 à 4, valide sur le
fold 0 et écrit results/baselines/fold0.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import compute_metrics  # noqa: E402

TRAIN_VAL = ROOT / "data/processed/train_val.csv"
OUTPUT = ROOT / "results/baselines/fold0.json"
VAL_FOLD = 0
SEED = 42


def extract_features(image_path):
    """Extrait 30 features couleur (moyennes, écarts-types, histogrammes 8 bins)."""
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    pixels = pixels.reshape(-1, 3)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, channel], bins=8, range=(0.0, 1.0))
        histograms.append(hist / pixels.shape[0])

    features = np.concatenate([means, stds, *histograms]).astype(np.float32)
    assert features.shape == (30,)
    return features


def main():
    train_val = pd.read_csv(TRAIN_VAL)
    assert (train_val["split"] == "train_val").all()
    train_df = train_val[train_val["fold"] != VAL_FOLD]
    val_df = train_val[train_val["fold"] == VAL_FOLD]

    image_paths = {
        path.stem: path
        for part in ("HAM10000_images_part_1", "HAM10000_images_part_2")
        for path in (ROOT / "data/raw" / part).glob("*.jpg")
    }
    missing = set(train_val["image_id"]) - image_paths.keys()
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) introuvable(s)")

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(train_df), 1)), y_train)
    dummy_metrics = compute_metrics(y_val, dummy.predict(np.zeros((len(val_df), 1))))

    def features(df):
        return np.vstack([extract_features(image_paths[i]) for i in df["image_id"]])

    color_model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
    )
    color_model.fit(features(train_df), y_train)
    logreg_metrics = compute_metrics(y_val, color_model.predict(features(val_df)))

    results = {
        "val_fold": VAL_FOLD,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "dummy": {k: float(v) for k, v in dummy_metrics.items()},
        "logreg": {k: float(v) for k, v in logreg_metrics.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
