"""Entraîne un ResNet-18 sur un fold et écrit tous les résultats du lancement.

    python -m scripts.train --config configs/resnet18_head.yaml --fold 0 --seed 0

Écrit dans runs/<nom_config>/fold<k>_seed<s>/ :
config.yaml, history.csv, best.pt, val_logits.csv, metrics.json.
Seul train_val.csv est lu : le test n'est jamais chargé.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classes import NUM_CLASSES  # noqa: E402
from src.dataset import DermDataset  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import evaluate, train_one_epoch  # noqa: E402
from src.transforms import get_train_transforms, get_val_transforms  # noqa: E402
from src.utils import get_class_weights, save_model, seed_worker, set_seed  # noqa: E402

METRIC_KEYS = ("loss", "accuracy", "balanced_accuracy", "f1")


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_optimizer(cfg, model):
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg["name"].lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    raise ValueError(f"Optimiseur inconnu : {cfg['name']}")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def run(cfg, fold, seed, *, runs_dir, train_val_path, image_dir=None, limit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = set_seed(seed)

    train_val = pd.read_csv(train_val_path)
    assert (train_val["split"] == "train_val").all(), "le test ne doit jamais être chargé"
    train_df = train_val[train_val["fold"] != fold].reset_index(drop=True)
    val_df = train_val[train_val["fold"] == fold].reset_index(drop=True)
    assert len(val_df) > 0, f"fold {fold} introuvable"
    assert set(train_df["lesion_id"]).isdisjoint(val_df["lesion_id"])
    if limit:  # lancement de contrôle uniquement
        train_df = train_df.sample(min(limit, len(train_df)), random_state=seed).reset_index(drop=True)
        val_df = val_df.sample(min(limit, len(val_df)), random_state=seed).reset_index(drop=True)

    image_dir = resolve(image_dir or cfg["image_dir"])
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"].get("num_workers", 2)
    train_tf = get_train_transforms() if cfg.get("augment", True) else get_val_transforms()
    train_loader = DataLoader(
        DermDataset(train_df, image_dir, transform=train_tf), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=generator,
    )
    val_loader = DataLoader(
        DermDataset(val_df, image_dir, transform=get_val_transforms()), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker,
    )

    m = cfg["model"]
    model = build_model(m["arch"], pretrained=m["pretrained"], trainable=m["trainable"]).to(device)
    optimizer = build_optimizer(cfg["optim"], model)
    weight = get_class_weights(train_df, NUM_CLASSES).to(device) if cfg["loss"]["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weight)

    out_dir = Path(runs_dir) / cfg["name"] / f"fold{fold}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best = [], None
    best_f1, patience_left = -1.0, cfg["train"]["patience"]
    for epoch in range(cfg["train"]["epochs"]):
        start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer=optimizer, criterion=criterion, device=device
        )
        metrics = evaluate(model, val_loader, criterion, device)
        duration = time.perf_counter() - start
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": metrics["loss"],
            "val_f1": metrics["f1"], "val_accuracy": metrics["accuracy"],
            "val_balanced_accuracy": metrics["balanced_accuracy"], "duration_s": duration,
        })
        print(f"epoch {epoch:2d} | train {train_loss:.4f} | val {metrics['loss']:.4f} "
              f"| f1 {metrics['f1']:.4f} | {duration:.0f} s", flush=True)

        if metrics["f1"] > best_f1:
            best_f1, patience_left = metrics["f1"], cfg["train"]["patience"]
            best = {"epoch": epoch, **metrics}
            save_model(model, out_dir / "best.pt",
                       {"fold": fold, "seed": seed, "epoch": epoch, "val_f1": metrics["f1"]})
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("arrêt précoce")
                break

    # Fichiers du lancement : tous issus de la meilleure époque.
    full_cfg = {**cfg, "fold": fold, "seed": seed, "git_commit": git_commit(),
                "image_dir": str(image_dir), "limit": limit}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(full_cfg, sort_keys=False, allow_unicode=True))
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    logits = pd.DataFrame(best["logits"].numpy(), columns=[f"logit_{i}" for i in range(NUM_CLASSES)])
    logits.insert(0, "label", best["labels"].numpy())
    logits.insert(0, "lesion_id", val_df["lesion_id"].values)
    logits.insert(0, "image_id", val_df["image_id"].values)
    logits.to_csv(out_dir / "val_logits.csv", index=False)

    result = {"fold": fold, "seed": seed, "best_epoch": best["epoch"],
              "epochs_run": len(history), "n_val": len(val_df),
              **{k: float(best[k]) for k in METRIC_KEYS}}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-dir", default=ROOT / "runs")
    parser.add_argument("--train-val", default=ROOT / "data/processed/train_val.csv")
    parser.add_argument("--image-dir", default=None, help="remplace image_dir de la config")
    parser.add_argument("--limit", type=int, default=None,
                        help="contrôle rapide : n images par partition (résultats non valides)")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    return run(cfg, args.fold, args.seed, runs_dir=args.runs_dir,
               train_val_path=args.train_val, image_dir=args.image_dir, limit=args.limit)


if __name__ == "__main__":
    main()
