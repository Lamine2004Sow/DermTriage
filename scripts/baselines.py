"""Baselines (DummyClassifier et régression logistique sur features couleur).

Lit data/processed/train_val.csv (le test n'est jamais chargé). Pour chaque fold k, entraîne sur les autres
folds, valide sur le fold k et écrit results/baselines/fold{k}.json.
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
OUTPUT_DIR = ROOT / "results/baselines"
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

    image_paths = {
        path.stem: path
        for part in ("HAM10000_images_part_1", "HAM10000_images_part_2")
        for path in (ROOT / "data/raw" / part).glob("*.jpg")
    }
    missing = set(train_val["image_id"]) - image_paths.keys()
    if missing:
        raise FileNotFoundError(f"{len(missing)} image(s) introuvable(s)")

    all_features = np.vstack([extract_features(image_paths[i]) for i in train_val["image_id"]])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for val_fold in sorted(train_val["fold"].unique()):
        is_val = (train_val["fold"] == val_fold).to_numpy()
        y_train = train_val.loc[~is_val, "label"].to_numpy()
        y_val = train_val.loc[is_val, "label"].to_numpy()

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(y_train), 1)), y_train)
        dummy_metrics = compute_metrics(y_val, dummy.predict(np.zeros((len(y_val), 1))))

        color_model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        )
        color_model.fit(all_features[~is_val], y_train)
        logreg_metrics = compute_metrics(y_val, color_model.predict(all_features[is_val]))

        results = {
            "val_fold": int(val_fold),
            "n_train": len(y_train),
            "n_val": len(y_val),
            "dummy": {k: float(v) for k, v in dummy_metrics.items()},
            "logreg": {k: float(v) for k, v in logreg_metrics.items()},
        }
        (OUTPUT_DIR / f"fold{val_fold}.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
