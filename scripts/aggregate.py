"""Agrège tous les résultats de la phase 3 dans results/comparison.csv.

    python -m scripts.aggregate [--results results] [--out results/comparison.csv]

Une ligne par variante, chacune comparée à la précédente du plan d'expériences (un seul changement) :
moyenne et écart-type de la macro-F1 sur les folds, écart apparié à la variante de référence et nombre de
folds gagnés. Quand une variante a plusieurs graines, les métriques d'un fold sont d'abord moyennées sur les
graines (la comparaison reste appariée fold par fold) ; `seed_std` est l'écart-type entre les moyennes des
graines. Les colonnes rappel/précision viennent des val_logits.csv (absents des baselines).
Seules les lectures de results/ sont faites : aucun entraînement, le test n'est jamais lu.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDS = range(5)
METRICS = ("f1", "balanced_accuracy", "accuracy")
# (id, description, dossier ou clé de baseline, référence, changement par rapport à la référence)
PLAN = [
    ("B0", "Classe majoritaire", "baseline:dummy", None, ""),
    ("B1", "Régression logistique, 30 features couleur", "baseline:logreg", "B0", "features couleur"),
    ("E1", "ResNet-18 figé + régression logistique", "resnet18_frozen_logreg", "B1", "représentation ImageNet"),
    ("E2", "ResNet-18, tête seule", "resnet18_head", "E1", "tête entraînée, augmentations"),
    ("E3", "layer4 + tête", "resnet18_layer4_rerun", "E2", "fine-tuning de layer4"),
    ("E4", "layer4 + tête, perte pondérée", "resnet18_layer4_weighted", "E3", "poids de classes"),
    ("E5", "layer4 + tête, sans augmentations", "resnet18_layer4_noaug", "E3", "sans augmentations"),
]
# indices des classes : akiec 0, df 3, nv 5, vasc 6
LOGIT_COLUMNS = ("recall_akiec", "recall_df", "recall_vasc", "precision_nv")


def seeds_available(variant_dir):
    """Graines pour lesquelles les 5 folds existent."""
    seeds = None
    for k in FOLDS:
        found = {int(m.group(1)) for p in variant_dir.glob(f"fold{k}_seed*")
                 if (m := re.fullmatch(rf"fold{k}_seed(\d+)", p.name)) and (p / "metrics.json").is_file()}
        seeds = found if seeds is None else seeds & found
    return sorted(seeds)


def logit_stats(path):
    d = pd.read_csv(path)
    pred = d[[f"logit_{i}" for i in range(7)]].to_numpy().argmax(axis=1)
    y = d["label"].to_numpy()

    def recall(c):
        return ((pred == c) & (y == c)).sum() / (y == c).sum()

    n_pred = (pred == 5).sum()
    precision_nv = ((pred == 5) & (y == 5)).sum() / n_pred if n_pred else 0.0
    return [recall(0), recall(3), recall(6), precision_nv]


def load_variant(results, source):
    """Renvoie (métriques[graine][fold][métrique], logits[graine][fold][4] ou None, graines)."""
    if source.startswith("baseline:"):
        key = source.split(":", 1)[1]
        data = [json.loads((results / "baselines" / f"fold{k}.json").read_text())[key] for k in FOLDS]
        return np.array([[[m[name] for name in METRICS] for m in data]]), None, [0]

    variant_dir = results / source
    seeds = seeds_available(variant_dir)
    if not seeds:
        raise FileNotFoundError(f"aucune graine complète (5 folds) dans {variant_dir}")
    metrics = np.array([[[json.loads((variant_dir / f"fold{k}_seed{s}" / "metrics.json").read_text())[name]
                          for name in METRICS] for k in FOLDS] for s in seeds])
    has_logits = all((variant_dir / f"fold{k}_seed{s}" / "val_logits.csv").is_file() for s in seeds for k in FOLDS)
    logits = None
    if has_logits:
        logits = np.array([[logit_stats(variant_dir / f"fold{k}_seed{s}" / "val_logits.csv") for k in FOLDS]
                           for s in seeds])
    return metrics, logits, seeds


def aggregate(results):
    results = Path(results)
    loaded = {row[0]: load_variant(results, row[2]) for row in PLAN}
    per_fold_f1 = {code: m[0][:, :, 0].mean(axis=0) for code, m in loaded.items()}

    rows = []
    for code, description, source, ref, change in PLAN:
        metrics, logits, seeds = loaded[code]
        by_fold = metrics.mean(axis=0)  # moyenne sur les graines : (folds, métriques)
        f1 = by_fold[:, 0]
        row = {
            "id": code, "variante": description, "dossier": source, "change_par_rapport_a": ref or "",
            "changement": change, "n_folds": len(f1), "n_graines": len(seeds),
            "f1_moyenne": f1.mean(), "f1_ecart_type_folds": f1.std(ddof=1),
            "f1_ecart_type_graines": np.std(metrics[:, :, 0].mean(axis=1), ddof=1) if len(seeds) > 1 else np.nan,
            "balanced_accuracy_moyenne": by_fold[:, 1].mean(), "accuracy_moyenne": by_fold[:, 2].mean(),
        }
        if ref:
            diff = f1 - per_fold_f1[ref]
            row.update({"ecart_moyen_f1": diff.mean(), "folds_gagnes": int((diff > 0).sum()),
                        "gagne_regle_phase_1": bool(diff.mean() > 0 and (diff > 0).sum() >= 4)})
        else:
            row.update({"ecart_moyen_f1": np.nan, "folds_gagnes": np.nan, "gagne_regle_phase_1": np.nan})
        stats = logits.mean(axis=(0, 1)) if logits is not None else [np.nan] * 4
        row.update(dict(zip(LOGIT_COLUMNS, stats)))
        rows.append(row)
    table = pd.DataFrame(rows)
    table["folds_gagnes"] = table["folds_gagnes"].astype("Int64")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=ROOT / "results")
    parser.add_argument("--out", default=None, help="par défaut <results>/comparison.csv")
    args = parser.parse_args(argv)
    table = aggregate(args.results)
    out = Path(args.out) if args.out else Path(args.results) / "comparison.csv"
    table.to_csv(out, index=False, float_format="%.6f")
    shown = table[["id", "n_graines", "f1_moyenne", "f1_ecart_type_folds", "ecart_moyen_f1", "folds_gagnes"]]
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"écrit {out}")
    return table


if __name__ == "__main__":
    main()
