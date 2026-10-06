"""E1 : ResNet-18 figé (poids ImageNet) comme extracteur de caractéristiques + régression logistique.

    python -m scripts.extract_features [--image-dir ...] [--train-val ...] [--C 1.0]

Le réseau est figé et les images ne sont pas augmentées : la sortie de 512 dimensions de chaque image est
toujours la même. Elle est donc calculée une seule fois (data/features/resnet18_imagenet.npz), puis une
régression logistique est entraînée sur chaque fold (les 4 autres folds pour apprendre, le fold k pour valider).

Écrit results/resnet18_frozen_logreg/fold<k>_seed0/metrics.json (même format que les runs d'entraînement).
Seul train_val.csv est lu : le test n'est jamais chargé.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import DermDataset  # noqa: E402
from src.models import build_model  # noqa: E402
from src.transforms import get_val_transforms  # noqa: E402
from src.utils import compute_metrics  # noqa: E402

NAME = "resnet18_frozen_logreg"
SEED = 42


def build_encoder(pretrained=True):
    """ResNet-18 sans couche de classification : renvoie le vecteur de 512 dimensions."""
    model = build_model("resnet18", pretrained=pretrained, trainable="head")
    model.fc = nn.Identity()
    return model.eval()


@torch.no_grad()
def extract(encoder, loader, device):
    encoder.to(device).eval()
    features = [encoder(images.to(device)).cpu() for images, _ in loader]
    return torch.cat(features).numpy().astype(np.float32)


def load_or_compute_features(train_val, image_dir, cache, *, pretrained, batch_size, num_workers, force):
    ids = train_val["image_id"].to_numpy().astype(str)
    if cache.is_file() and not force:
        saved = np.load(cache, allow_pickle=False)
        if np.array_equal(saved["image_id"], ids):
            print(f"Caractéristiques lues dans {cache}")
            return saved["features"]
        print("Cache incompatible avec train_val.csv : recalcul")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        DermDataset(train_val, image_dir, transform=get_val_transforms()),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    features = extract(build_encoder(pretrained), loader, device)
    assert features.shape == (len(train_val), 512)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, image_id=ids, features=features)
    print(f"Caractéristiques {features.shape} écrites dans {cache}")
    return features


def evaluate_folds(train_val, features, *, C, out_dir):
    labels = train_val["label"].to_numpy()
    folds = train_val["fold"].to_numpy()
    results = []
    for fold in sorted(np.unique(folds)):
        is_val = folds == fold
        assert set(train_val.loc[~is_val, "lesion_id"]).isdisjoint(train_val.loc[is_val, "lesion_id"])
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=1000, random_state=SEED)
        )
        model.fit(features[~is_val], labels[~is_val])
        metrics = {k: float(v) for k, v in compute_metrics(labels[is_val], model.predict(features[is_val])).items()}
        result = {"fold": int(fold), "seed": 0, "n_train": int((~is_val).sum()), "n_val": int(is_val.sum()),
                  "C": C, **metrics}
        fold_dir = Path(out_dir) / f"fold{fold}_seed0"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"fold {fold} : macro-F1 {metrics['f1']:.4f}")
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-val", default=ROOT / "data/processed/train_val.csv")
    parser.add_argument("--image-dir", default=ROOT / "data/interim/256")
    parser.add_argument("--cache", default=ROOT / "data/features/resnet18_imagenet.npz")
    parser.add_argument("--out-dir", default=ROOT / "results" / NAME)
    parser.add_argument("--C", type=float, default=1.0, help="inverse de la régularisation (fixé, non optimisé)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="recalcule les caractéristiques")
    parser.add_argument("--no-pretrained", action="store_true", help="poids aléatoires (tests uniquement)")
    args = parser.parse_args(argv)

    train_val = pd.read_csv(args.train_val)
    assert (train_val["split"] == "train_val").all(), "le test ne doit jamais être chargé"
    features = load_or_compute_features(
        train_val, args.image_dir, Path(args.cache), pretrained=not args.no_pretrained,
        batch_size=args.batch_size, num_workers=args.num_workers, force=args.force,
    )
    return evaluate_folds(train_val, features, C=args.C, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
