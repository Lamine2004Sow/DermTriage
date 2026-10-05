# Phase 0 — Changements de code

Branche `refactor/phase-0-assainissement`.

| Élément | Changement | Pourquoi |
|---|---|---|
| `src/classes.py` | `CLASSES` figé (+ `NUM_CLASSES`, `CLASS_TO_INDEX`) | Un label ne dépend plus des données présentes |
| `scripts/make_splits.py` | Logique de `exploration.ipynb` en script | Partitions reproductibles, même images pour tous les modèles |
| `tests/test_splits.py` | 6 tests sur le vrai fichier, skip si absent | Garantir absence de fuite et classes dans chaque fold |
| `get_class_weights` | `.reindex(range(num_classes), fill_value=0)` | Une classe absente décalait les poids |
| `get_class_weights` | Effectif nul → poids 0 | Évite l'infini ; aucun exemple n'utilise cette classe |
| `train_one_epoch` | `*` : `optimizer`, `criterion`, `device` nommés | Interdit l'inversion silencieuse d'arguments |
| `requirements.txt` | torch, torchvision, numpy, pandas, scikit-learn, pillow, matplotlib, pyyaml | Dépendances réelles, sans freeze |
| `requirements-dev.txt` | `-r requirements.txt`, pytest, jupyterlab | Outils de développement séparés |
| `scripts/preprocess.py` | Petit côté à 256 px dans `data/interim/256/` | Éviter de décoder un JPEG 600×450 à chaque époque |
| `scripts/baselines.py` | Remplace `baseline.ipynb`, écrit `fold0.json` | Résultat chiffré reproductible |
| `Makefile` | cibles `install`, `splits`, `preprocess`, `test`, `baselines` | Quatre commandes pour tout refaire |

## Correctif `get_class_weights` (TDD)
Test écrit d'abord (`test_class_weights_classe_absente`) : classe 3 absente,
poids attendus finis, `weights[3] == 0`, `weights[6]` et `weights[5]` alignés sur
leur indice. Il échouait avant le correctif (ancien code : 6 poids au lieu de 7).

## Tests
38 tests passent (`make test`).

## Compléments
| Élément | Changement | Pourquoi |
|---|---|---|
| `scripts/baselines.py` | Lit `train_val.csv` | Le test n'est jamais chargé ; résultats inchangés (`fold0.json` identique) |
| `src/dataset.py` | Accepte aussi un dossier plat (`data/interim/256`) | Utiliser les images pré-redimensionnées sans changer le contrat |
| `tests/test_dataset.py` | Test du dossier plat | 39 tests passent |
| `make preprocess` | 10 015 images à 256 px, 257 Mo, ≈ 5 min | Réduire le coût de décodage JPEG |

Temps d'une époque : voir `04_calcul.md`.
