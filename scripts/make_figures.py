"""Figures de la phase 3, reconstruites uniquement à partir de results/ (aucun entraînement).

    python -m scripts.make_figures [--out figures]

Sources : results/baselines/fold*.json, results/<variante>/fold<k>_seed0/{metrics.json, history.csv, val_logits.csv}.
E3 est la version relancée (resnet18_layer4_rerun) : c'est la seule qui a ses val_logits.csv, dans la même
session Kaggle que E4 et E5. Seed 0 uniquement ; les graines 1 et 2 s'ajouteront à part.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FOLDS = range(5)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Couleur par entité (jamais par rang) : ordre fixe des slots catégoriels.
VARIANTS = {
    "E1": ("ResNet figé + logreg", "resnet18_frozen_logreg", "#2a78d6"),
    "E2": ("Tête seule", "resnet18_head", "#eb6834"),
    "E3": ("layer4 + tête", "resnet18_layer4_rerun", "#1baf7a"),
    "E4": ("layer4, perte pondérée", "resnet18_layer4_weighted", "#eda100"),
    "E5": ("layer4, sans augmentations", "resnet18_layer4_noaug", "#e87ba4"),
}
BASE_GREY = "#8a8985"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
    })


def load_metrics(code):
    d = RESULTS / VARIANTS[code][1]
    return [json.loads((d / f"fold{k}_seed0" / "metrics.json").read_text()) for k in FOLDS]


def load_history(code, fold):
    return pd.read_csv(RESULTS / VARIANTS[code][1] / f"fold{fold}_seed0" / "history.csv")


def load_logits(code, fold):
    d = pd.read_csv(RESULTS / VARIANTS[code][1] / f"fold{fold}_seed0" / "val_logits.csv")
    scores = d[[f"logit_{i}" for i in range(7)]].to_numpy()
    return d["label"].to_numpy(), scores.argmax(axis=1)


def pooled_predictions(code):
    pairs = [load_logits(code, k) for k in FOLDS]
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


def baseline_f1(model):
    return [json.loads((RESULTS / "baselines" / f"fold{k}.json").read_text())[model]["f1"] for k in FOLDS]


def f1_by_fold():
    out = {"B0": baseline_f1("dummy"), "B1": baseline_f1("logreg")}
    for code in VARIANTS:
        out[code] = [m["f1"] for m in load_metrics(code)]
    return out


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print("écrit", out / name)


def fig_macro_f1(out):
    f1 = f1_by_fold()
    rows = [("B0", "Classe majoritaire", BASE_GREY), ("B1", "Logreg sur couleurs", BASE_GREY)] + [
        (c, f"{c} · {VARIANTS[c][0]}", VARIANTS[c][2]) for c in VARIANTS]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (code, label, color) in enumerate(rows):
        y = len(rows) - 1 - i
        v = np.array(f1[code])
        ax.plot([v.min(), v.max()], [y, y], color=color, lw=2, alpha=0.5, solid_capstyle="round", zorder=2)
        ax.scatter(v, [y] * len(v), s=46, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.plot([v.mean()] * 2, [y - 0.28, y + 0.28], color=INK, lw=2, zorder=4)
        ax.text(0.865, y, f"{v.mean():.3f}", va="center", ha="right", color=INK, fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows][::-1])
    ax.set_xlim(0, 0.87)
    ax.set_xlabel("Macro-F1 de validation (un point par fold, trait noir = moyenne)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Macro-F1 par variante, 5 folds, seed 0", pad=10)
    save(fig, out, "fig1_macro_f1_par_variante.png")


def fig_paired_differences(out):
    f1 = f1_by_fold()
    panels = [("H1 · E1 contre B1", "E1", "B1", "E1"), ("H2 · E3 contre E2", "E3", "E2", "E3"),
              ("H4 · E3 contre E5", "E3", "E5", "E3")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, (title, a, b, color_code) in zip(axes, panels):
        diff = np.array(f1[a]) - np.array(f1[b])
        color = VARIANTS[color_code][2]
        ax.bar(list(FOLDS), diff, color=color, width=0.62)
        ax.axhline(0, color=INK2, lw=1)
        for k, d in zip(FOLDS, diff):
            ax.text(k, d + (0.006 if d >= 0 else -0.006), f"{d:+.3f}", ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=8.5, color=INK)
        wins = int((diff > 0).sum())
        ax.set_title(f"{title}\n{wins}/5 folds · moy. {diff.mean():+.3f}", fontsize=10)
        ax.set_xticks(list(FOLDS))
        ax.set_xlabel("Fold")
        ax.grid(axis="x", visible=False)
        lim = max(abs(diff).max() * 1.35, 0.05)
        ax.set_ylim(-lim if diff.min() < 0 else -lim * 0.1, lim)
    axes[0].set_ylabel("Écart de macro-F1")
    fig.suptitle("Comparaisons appariées (macro-F1)", x=0.01, ha="left", fontweight="bold", y=1.04)
    save(fig, out, "fig2_ecarts_apparies_H1_H2_H4.png")


def class_stats(code):
    y, p = pooled_predictions(code)
    recall = [((p == i) & (y == i)).sum() / (y == i).sum() for i in range(7)]
    precision = [((p == i) & (y == i)).sum() / max((p == i).sum(), 1) for i in range(7)]
    return np.array(recall), np.array(precision), np.bincount(y, minlength=7)


def per_fold_class_stats(code):
    rec, prec = [], []
    for k in FOLDS:
        y, p = load_logits(code, k)
        rec.append([((p == i) & (y == i)).sum() / (y == i).sum() for i in range(7)])
        prec.append([((p == i) & (y == i)).sum() / max((p == i).sum(), 1) for i in range(7)])
    return np.array(rec), np.array(prec)


def fig_h3(out):
    r3, p3 = per_fold_class_stats("E3")
    r4, p4 = per_fold_class_stats("E4")
    crit = [("Rappel akiec", r4[:, 0] - r3[:, 0]), ("Rappel df", r4[:, 3] - r3[:, 3]),
            ("Rappel vasc", r4[:, 6] - r3[:, 6]), ("Précision nv", p4[:, 5] - p3[:, 5])]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    color = VARIANTS["E4"][2]
    for ax, (title, diff) in zip(axes, crit):
        ax.bar(list(FOLDS), diff, color=color, width=0.62)
        ax.axhline(0, color=INK2, lw=1)
        for k, d in zip(FOLDS, diff):
            ax.text(k, d + (0.008 if d >= 0 else -0.008), f"{d:+.2f}", ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=8, color=INK)
        ax.set_title(f"{title}\n{int((diff > 0).sum())} sur 5 · moy. {diff.mean():+.3f}", fontsize=10)
        ax.set_xticks(list(FOLDS))
        ax.set_xlabel("Fold")
        ax.grid(axis="x", visible=False)
        lim = max(abs(diff).max() * 1.4, 0.05)
        ax.set_ylim(-lim if diff.min() < 0 else -lim * 0.1, lim)
    axes[0].set_ylabel("E4 − E3")
    fig.suptitle("H3 · effet de la perte pondérée sur les critères du protocole", x=0.01, ha="left",
                 fontweight="bold", y=1.04)
    save(fig, out, "fig3_H3_classes_rares.png")


def fig_learning_curves_f1(out):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, code in zip(axes, ["E2", "E3", "E4", "E5"]):
        color = VARIANTS[code][2]
        for k in FOLDS:
            h = load_history(code, k)
            ax.plot(h["epoch"], h["val_f1"], color=color, lw=1.6, alpha=0.8)
            best = int(h["val_f1"].idxmax())
            ax.scatter([best], [h["val_f1"].iloc[best]], s=26, color=color, edgecolor=SURFACE, zorder=3)
        ax.set_title(f"{code} · {VARIANTS[code][0]}", fontsize=10)
        ax.set_xlabel("Époque")
        ax.set_xlim(-0.5, 20)
    axes[0].set_ylabel("Macro-F1 de validation")
    fig.suptitle("Courbes d'apprentissage (une courbe par fold, point = meilleure époque)", x=0.01,
                 ha="left", fontweight="bold", y=1.04)
    save(fig, out, "fig4_courbes_macro_f1.png")


def fig_loss_e3(out):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=True)
    for k, ax in zip(FOLDS, axes):
        h = load_history("E3", k)
        best = int(json.loads((RESULTS / VARIANTS["E3"][1] / f"fold{k}_seed0" / "metrics.json").read_text())["best_epoch"])
        ax.plot(h["epoch"], h["train_loss"], color="#2a78d6", lw=2, label="Entraînement")
        ax.plot(h["epoch"], h["val_loss"], color="#eb6834", lw=2, label="Validation")
        ax.axvline(best, color=INK2, lw=1, ls=":")
        ax.set_title(f"Fold {k} · époque {best}", fontsize=10)
        ax.set_xlabel("Époque")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel("Perte")
    axes[0].legend(loc="upper right")
    fig.suptitle("E3 · perte d'entraînement et de validation (pointillé = époque retenue)", x=0.01,
                 ha="left", fontweight="bold", y=1.05)
    save(fig, out, "fig5_courbes_perte_E3.png")


def fig_confusions(out):
    for code in ["E3", "E4", "E5", "E2", "E1"]:
        y, p = pooled_predictions(code)
        cm = np.zeros((7, 7))
        for t, q in zip(y, p):
            cm[t, q] += 1
        norm = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("bleu", BLUE_RAMP)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if norm[i, j] > 0.5 else INK)
        n = cm.sum(axis=1).astype(int)
        ax.set_xticks(range(7))
        ax.set_xticklabels(CLASSES)
        ax.set_yticks(range(7))
        ax.set_yticklabels([f"{c} (n={n[i]})" for i, c in enumerate(CLASSES)])
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{code} · {VARIANTS[code][0]}\nMatrice de confusion (lignes = rappel), 5 folds cumulés", fontsize=10)
        save(fig, out, f"fig6_confusion_{code}.png")


def fig_recall_by_class(out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, which, title in zip(axes, (0, 1), ("Rappel par classe", "Précision par classe")):
        n = None
        for code in VARIANTS:
            rec, prec, n = class_stats(code)
            vals = (rec, prec)[which]
            ax.plot(vals, range(7), "o", color=VARIANTS[code][2], ms=7, mec=SURFACE, mew=1.2,
                    label=f"{code} · {VARIANTS[code][0]}", zorder=3)
        ax.set_yticks(range(7))
        ax.set_yticklabels([f"{c} (n={n[i]})" for i, c in enumerate(CLASSES)])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel(title + " (5 folds cumulés)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[1].set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    save(fig, out, "fig7_rappel_precision_par_classe.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=ROOT / "figures")
    args = parser.parse_args(argv)
    style()
    out = Path(args.out)
    for fn in (fig_macro_f1, fig_paired_differences, fig_h3, fig_learning_curves_f1, fig_loss_e3,
               fig_confusions, fig_recall_by_class):
        fn(out)


if __name__ == "__main__":
    main()
