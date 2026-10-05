# Phase 0 — Splits reproductibles

**Commande :** `make splits` → `scripts/make_splits.py`
**Sorties (`data/processed/`) :** `splits.csv` (tout), `train_val.csv`, `test.csv`
**Colonnes :** `image_id, lesion_id, dx, label, fold, split`

## Méthode
- Source : `data/raw/HAM10000_metadata.csv` (10 015 images).
- `StratifiedGroupKFold` : stratifié par classe `dx`, groupé par `lesion_id`
  (une lésion peut avoir plusieurs images → pas de fuite entre partitions).
- Étape 1 : 10 splits, 1 retenu → test (~10 %, `fold = -1`), `random_state=42`.
- Étape 2 : le reste → 5 folds (0 à 4), `random_state=42`.
- Mapping classe → entier figé dans `src/classes.py` :
  `CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]`
  (plus de `LabelEncoder` ni de `sorted(unique())`). Il coïncide avec l'ancien
  ordre alphabétique : aucun label n'a changé.
- `train_val.csv` et `test.csv` sont écrits séparément pour ne pas toucher au
  test par accident en lisant `splits.csv`.

## Vérification
Régénéré avec `make splits` : identique à l'ancien `splits.csv` du notebook
(hors colonne `label`, ajoutée). Le split est donc bien reproductible.

## Tailles
| Partition | Images | Lésions |
|---|---:|---:|
| test | 1 002 | 747 |
| train_val | 9 013 | 6 723 |
| fold 0 | 1 803 | 1 345 |
| fold 1 | 1 803 | 1 345 |
| fold 2 | 1 803 | 1 345 |
| fold 3 | 1 802 | 1 344 |
| fold 4 | 1 802 | 1 344 |

## Images par classe et par partition
| Classe | test | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| akiec | 32 | 59 | 59 | 59 | 59 | 59 | 327 |
| bcc | 51 | 92 | 93 | 92 | 93 | 93 | 514 |
| bkl | 110 | 198 | 198 | 197 | 198 | 198 | 1 099 |
| df | 12 | 21 | 21 | 21 | 20 | 20 | 115 |
| mel | 112 | 200 | 200 | 201 | 200 | 200 | 1 113 |
| nv | 671 | 1 207 | 1 206 | 1 207 | 1 207 | 1 207 | 6 705 |
| vasc | 14 | 26 | 26 | 26 | 25 | 25 | 142 |

`nv` ≈ 67 % des images : l'accuracy seule est trompeuse, d'où l'usage de la
balanced accuracy et du F1 macro.

## Tests (`tests/test_splits.py`, ignorés si `splits.csv` absent)
- aucune lésion commune entre `train_val` et `test` ;
- chaque lésion dans un seul fold ;
- les 7 classes présentes dans chaque fold ;
- `label == CLASSES.index(dx)` ;
- colonnes attendues, `image_id` uniques.

## Reproductibilité entre machines
Rejoué en local après suppression de `data/processed/` : fichiers identiques
(MD5 : voir `04_calcul.md`). Sur Kaggle, d'autres versions de bibliothèques donnent
d'autres partitions : les CSV doivent être copiés, pas régénérés.
