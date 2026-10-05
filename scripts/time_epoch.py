"""Chronomètre une époque d'entraînement ResNet-18 (folds 1 à 4 de train_val.csv).

Exemples :
    python scripts/time_epoch.py                       # CPU ou GPU détecté
    python scripts/time_epoch.py --max-batches 20      # extrapolation rapide
    python scripts/time_epoch.py --data-dir /kaggle/input/dermtriage-raw
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classes import NUM_CLASSES  # noqa: E402
from src.dataset import DermDataset  # noqa: E402
from src.train import train_one_epoch  # noqa: E402
from src.transforms import get_train_transforms  # noqa: E402
from src.utils import get_class_weights  # noqa: E402

VAL_FOLD = 0


class LimitedLoader:
    """Limite un DataLoader à n lots pour mesurer sans parcourir l'époque entière."""

    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = [None] * min(len(loader.dataset), max_batches * loader.batch_size)

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                return
            yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=ROOT / "data/raw")
    parser.add_argument("--train-val", default=ROOT / "data/processed/train_val.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="mesure sur n lots puis extrapole à l'époque complète")
    parser.add_argument("--pretrained", action="store_true",
                        help="poids ImageNet (télécharge ~45 Mo) ; sans effet sur le temps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv(args.train_val)
    train_df = train_df[train_df["fold"] != VAL_FOLD]

    dataset = DermDataset(train_df, args.data_dir, transform=get_train_transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers)
    n_batches = len(loader)

    model = models.resnet18(weights="IMAGENET1K_V1" if args.pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(train_df).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    run_loader = LimitedLoader(loader, args.max_batches) if args.max_batches else loader
    measured = min(args.max_batches or n_batches, n_batches)

    print(f"device={device} | images={len(dataset)} | lots={n_batches} "
          f"(batch_size={args.batch_size}, workers={args.num_workers})")
    start = time.perf_counter()
    loss = train_one_epoch(model, run_loader, optimizer=optimizer,
                           criterion=criterion, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_batch = elapsed / measured
    print(f"loss={loss:.4f} | {measured} lot(s) en {elapsed:.1f} s "
          f"({per_batch:.3f} s/lot)")
    print(f"Époque complète : {per_batch * n_batches / 60:.1f} min"
          + (" (extrapolé)" if measured < n_batches else ""))


if __name__ == "__main__":
    main()
