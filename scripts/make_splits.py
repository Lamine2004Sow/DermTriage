"""Construit data/processed/splits.csv à partir des métadonnées HAM10000.

Colonnes : image_id, lesion_id, dx, label, fold, split.
- split "test" : ~10 % des images, fold = -1 ;
- split "train_val" : le reste, réparti en 5 folds (0 à 4).
Les partitions sont stratifiées par classe et groupées par lésion.
Le mapping dx -> label vient de src/classes.py.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classes import CLASS_TO_INDEX  # noqa: E402

SEED = 42
METADATA = ROOT / "data/raw/HAM10000_metadata.csv"
OUTPUT = ROOT / "data/processed/splits.csv"


def make_splits(metadata):
    unknown = set(metadata["dx"]) - set(CLASS_TO_INDEX)
    if unknown:
        raise ValueError(f"Classes inconnues : {sorted(unknown)}")

    # 1. Réserver environ 10 % des images pour le test final.
    test_splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    rest_idx, test_idx = next(
        test_splitter.split(metadata, y=metadata["dx"], groups=metadata["lesion_id"])
    )
    rest = metadata.iloc[rest_idx].copy()
    test = metadata.iloc[test_idx].copy()

    # 2. Attribuer un fold de validation croisée (0 à 4) au reste.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rest["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        cv.split(rest, y=rest["dx"], groups=rest["lesion_id"])
    ):
        rest.iloc[val_idx, rest.columns.get_loc("fold")] = fold
    rest["split"] = "train_val"

    test["fold"] = -1
    test["split"] = "test"

    splits = pd.concat([rest, test], ignore_index=True)
    splits["label"] = splits["dx"].map(CLASS_TO_INDEX).astype(int)
    return splits[["image_id", "lesion_id", "dx", "label", "fold", "split"]]


def main():
    metadata = pd.read_csv(METADATA)
    splits = make_splits(metadata)

    assert set(splits.loc[splits["split"] == "train_val", "lesion_id"]).isdisjoint(
        splits.loc[splits["split"] == "test", "lesion_id"]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(OUTPUT, index=False)

    print(f"{OUTPUT.relative_to(ROOT)} écrit : {len(splits)} images")
    print(splits.groupby(["split", "fold"]).size().to_string())


if __name__ == "__main__":
    main()
