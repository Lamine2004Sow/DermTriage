"""scripts/aggregate.py sur un faux dossier results/ (aucune donnée réelle)."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate import PLAN, aggregate, main, seeds_available


def write_run(path, f1, with_logits=True):
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({"f1": f1, "balanced_accuracy": f1 - 0.1, "accuracy": f1 + 0.1}))
    if with_logits:
        rng = np.random.default_rng(0)
        labels = np.arange(21) % 7
        logits = np.eye(7)[labels] * 5 + rng.normal(size=(21, 7)) * 0.1  # prédictions = labels
        d = pd.DataFrame(logits, columns=[f"logit_{i}" for i in range(7)])
        d.insert(0, "label", labels)
        d.to_csv(path / "val_logits.csv", index=False)


@pytest.fixture
def fake(tmp_path):
    for k in range(5):
        (tmp_path / "baselines").mkdir(exist_ok=True)
        (tmp_path / "baselines" / f"fold{k}.json").write_text(json.dumps({
            "dummy": {"f1": 0.1, "balanced_accuracy": 0.14, "accuracy": 0.67},
            "logreg": {"f1": 0.2 + 0.01 * k, "balanced_accuracy": 0.2, "accuracy": 0.66}}))
    levels = {"resnet18_frozen_logreg": 0.5, "resnet18_head": 0.55, "resnet18_layer4_rerun": 0.65,
              "resnet18_layer4_weighted": 0.6, "resnet18_layer4_noaug": 0.62}
    for name, base in levels.items():
        seeds = (0, 1) if name == "resnet18_layer4_weighted" else (0,)
        for s in seeds:
            for k in range(5):
                write_run(tmp_path / name / f"fold{k}_seed{s}", base + 0.01 * k + 0.02 * s)
    return tmp_path


def test_une_ligne_par_variante_du_plan(fake):
    table = aggregate(fake)
    assert list(table["id"]) == [row[0] for row in PLAN]


def test_moyenne_sur_les_graines_et_ecart_apparie(fake):
    t = aggregate(fake).set_index("id")
    # E4 : deux graines (0.6 + 0.01k) et (0.62 + 0.01k) -> moyenne par fold 0.61 + 0.01k
    assert t.loc["E4", "n_graines"] == 2
    assert t.loc["E4", "f1_moyenne"] == pytest.approx(0.63)
    assert t.loc["E4", "f1_ecart_type_graines"] == pytest.approx(0.02 / np.sqrt(2))
    # E4 contre E3 (0.65 + 0.01k) : écart constant de -0.04, aucun fold gagné
    assert t.loc["E4", "ecart_moyen_f1"] == pytest.approx(-0.04)
    assert t.loc["E4", "folds_gagnes"] == 0 and not t.loc["E4", "gagne_regle_phase_1"]
    # E3 contre E2 : +0.10 sur les 5 folds
    assert t.loc["E3", "folds_gagnes"] == 5 and t.loc["E3", "gagne_regle_phase_1"]


def test_colonnes_rappel_precision(fake):
    t = aggregate(fake).set_index("id")
    assert t.loc["E3", "recall_df"] == pytest.approx(1.0)
    assert t.loc["E3", "precision_nv"] == pytest.approx(1.0)
    assert np.isnan(t.loc["B1", "recall_df"])  # pas de logits pour les baselines


def test_graines_incompletes_ignorees(fake):
    # la graine 1 de E4 n'a que 4 folds : seule la graine 0 est retenue
    import shutil
    shutil.rmtree(fake / "resnet18_layer4_weighted" / "fold4_seed1")
    assert seeds_available(fake / "resnet18_layer4_weighted") == [0]


def test_ecrit_le_csv(fake):
    main(["--results", str(fake)])
    out = pd.read_csv(fake / "comparison.csv")
    assert len(out) == len(PLAN) and "f1_moyenne" in out.columns
