# Phase 0 — Budget de calcul (une époque ResNet-18)

**Commande :** `python scripts/time_epoch.py --num-workers 2`
(`make time-epoch` : 20 lots, extrapolé)

## Protocole
- ResNet-18 sans poids pré-entraînés, `fc` remplacée par 7 sorties.
- Entraînement sur les folds 1 à 4 : 7 210 images, 226 lots de 32.
- Transformations d'entraînement de `src/transforms.py` (Resize 224, flips,
  ColorJitter, normalisation), lecture des JPEG pleine résolution depuis `data/raw`.
- Adam, lr = 1e-4, loss pondérée par `get_class_weights`.
- Mesure : `time.perf_counter()` autour de `train_one_epoch`, avec synchronisation CUDA.

## Résultats
| Machine | Mesure | s / lot | Époque complète |
|---|---|---:|---:|
| Kaggle, Tesla T4 (2 workers) | époque entière, 226 lots en 73,8 s | 0,327 | **1,2 min** |
| Portable, CPU 8 cœurs (2 workers) | 8 lots, extrapolé | 4,68 | **≈ 17,6 min** |
| Portable, CPU 8 cœurs (2 workers), images 256 px | 8 lots, extrapolé | 3,87 | ≈ 14,6 min |
| Portable, CPU 8 cœurs (0 worker) | 3 lots, extrapolé | 4,86 | ≈ 18,3 min |

L'estimation du guide (10 à 20 min par époque sur CPU) est confirmée. Les temps
CPU sont des extrapolations sur quelques lots (démarrage inclus) : ordre de
grandeur, pas mesure précise.

## Conséquences pour le budget
- Le GPU T4 est environ **14 fois plus rapide** que le CPU : un fine-tuning de
  20 époques coûte ≈ 25 min sur Kaggle contre ≈ 6 h sur le portable.
- Les expériences d'entraînement se font sur Kaggle ; le CPU local sert au
  développement et aux tests.
- Pré-redimensionnement (`make preprocess`, `--data-dir data/interim/256`) : −17 %
  par lot sur CPU (4,68 → 3,87 s). Faible gain sur CPU, où le calcul du modèle
  domine ; il devrait compter davantage sur GPU, où le décodage JPEG peut
  devenir le goulot (non mesuré). Le cache de features (phase 3) reste un levier.

## Reproductibilité sur Kaggle
- Datasets utilisés : images HAM10000 (métadonnées identiques, MD5 `8f85fb1a…`)
  et `dermtriage-splits` (`splits.csv`, `train_val.csv`, `test.csv`).
- Régénérer les splits sur Kaggle donne des partitions **différentes**
  (MD5 `2f975107…` contre `132c1875…`) alors que les métadonnées sont identiques :
  la cause est la version de scikit-learn ou de numpy. Les splits locaux sont donc
  importés tels quels. MD5 de référence :
  `splits.csv` 132c187583dee168d04f11f39d737cb8,
  `train_val.csv` 641df133a9a2e40e9a35b66f224434b4,
  `test.csv` 35ce9feea522a7784999b36e65427132.
- Versions locales : scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3.
