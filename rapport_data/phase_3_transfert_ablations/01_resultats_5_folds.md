# Phase 3 — Expériences de transfert et ablations (en cours)

## Commandes
- Baselines : `make baselines` (`scripts/baselines.py`, 5 folds, ≈ 4,6 min sur CPU) → `results/baselines/fold{0..4}.json`.
- ResNet-18 tête seule (E2) : `python -m scripts.train --config configs/resnet18_head.yaml --fold k --seed 0`,
  k = 0 à 4, sur CPU local (≈ 196 s par époque), `runs/` copié vers `results/resnet18_head/fold{k}_seed0/`.
- Commit : `32e6c9e`.

## Résultats par fold (macro-F1, seed 0)
| Fold | Meilleure époque | Époques | Dummy (B0) | Logreg couleur (B1) | ResNet-18 tête (E2) | Balanced acc. (E2) | Accuracy (E2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14 | 19 | 0.1145704793545325 | 0.2435538079279549 | 0.5424631950301545 | 0.5262438430302644 | 0.747642817526345 |
| 1 | 5 | 10 | 0.1145136020509899 | 0.26984657821770075 | 0.45807036299314413 | 0.42442689224626406 | 0.7249029395452025 |
| 2 | 13 | 18 | 0.1145704793545325 | 0.2282608164513759 | 0.5179322347780302 | 0.4667792007041015 | 0.771491957848031 |
| 3 | 14 | 19 | 0.11460855528652139 | 0.26826889245519586 | 0.5185230720887791 | 0.47043039830670347 | 0.7602663706992231 |
| 4 | 6 | 11 | 0.11460855528652139 | 0.29046022291541895 | 0.49820753875572177 | 0.45467381311114313 | 0.7513873473917869 |

| | Moyenne | Écart-type (n-1) |
|---|---:|---:|
| Dummy | 0.11457433426661953 | 3.8923897273810224e-05 |
| Logreg couleur | 0.26007806359352925 | 0.024346341621846318 |
| ResNet-18 tête | 0.5070392807291659 | 0.031548938915708175 |

## Comparaison appariée (règle de la phase 1 : victoire sur ≥ 4 folds sur 5)
ResNet-18 tête contre logreg couleur : écarts de macro-F1 par fold
0.2989, 0.1882, 0.2897, 0.2503, 0.2077. **5 folds sur 5 gagnés**, écart moyen ≈ +0,247.
Ce résultat va dans le sens de H1, mais E1 (caractéristiques figées en cache + logreg), la
comparaison définie par le protocole, n'est pas encore faite ; la seed est unique.

## Observations
- Fold 1 : le plus faible (0,458), arrêt précoce à l'époque 10, meilleur modèle à l'époque 5.
- Accuracy ≈ 0,72 à 0,77 mais balanced accuracy ≈ 0,42 à 0,53 : les classes rares sont mal reconnues.

## E3 — ResNet-18 `layer4` + tête (5 folds, seed 0, Kaggle T4, commit `32e6c9e`)
Commande : `python -m scripts.train --config configs/resnet18_layer4.yaml --fold k --seed 0 --train-val <dermtriage-splits>/train_val.csv --image-dir <interim-256>/256`
(MD5 des splits vérifiés avant le lancement). Résultats : `results/resnet18_layer4/fold{k}_seed0/`.

| Fold | Meilleure époque | Époques | Macro-F1 E3 | Balanced acc. | Accuracy | Macro-F1 E2 | Écart E3 − E2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 11 | 0.6594127595657494 | 0.6359782694509907 | 0.8141985579589573 | 0.5424631950301545 | 0.11694956453559491 |
| 1 | 5 | 10 | 0.6494040396019376 | 0.605305606930541 | 0.8103161397670549 | 0.45807036299314413 | 0.19133367660879347 |
| 2 | 15 | 20 | 0.6516792886480159 | 0.6020401161281105 | 0.8180809761508597 | 0.5179322347780302 | 0.13374705386998575 |
| 3 | 6 | 11 | 0.6830852510742751 | 0.6691162389479522 | 0.8218645948945617 | 0.5185230720887791 | 0.16456217898549597 |
| 4 | 10 | 15 | 0.6757426438372786 | 0.661183413225703 | 0.8174250832408435 | 0.49820753875572177 | 0.1775351050815568 |

Moyenne E3 : 0.6638647965454513 (écart-type 0.014899075944073036). **5 folds sur 5 gagnés contre E2**, écart moyen ≈ +0,157 : H2 confirmée sur la seed 0 (règle de la phase 1).
Durée mesurée : ≈ 33 s par époque (moyenne hors première époque), 5 à 11 min par run, ≈ 42 min pour les 5 folds. Première époque du fold 0 : 267 s (démarrage).

## E3 relancé, E4 (perte pondérée), E5 (sans augmentations) — 5 folds, seed 0, Kaggle T4, commit `330a72c`
Configs : `resnet18_layer4_weighted.yaml` (`class_weights: true`), `resnet18_layer4_noaug.yaml` (`augment: false`).
Résultats : `results/resnet18_layer4_weighted/`, `results/resnet18_layer4_noaug/`, `results/resnet18_layer4_rerun/`
(E3 relancé dans la même session que E4 et E5, avec `val_logits.csv` conservés dans `runs/`, non versionné).

**Non-déterminisme entre sessions Kaggle.** E3 relancé (même config, même graine) ≠ E3 d'origine (macro-F1) :
fold 0 : 0.6594 → 0.6554 ; fold 1 : 0.6494 → 0.6472 ; fold 2 : 0.6517 → 0.6469 ; fold 3 : 0.6831 → 0.6802 ;
fold 4 : 0.6757 → 0.6748. Écart de 0,001 à 0,005, toujours à la baisse. Cause probable : non-déterminisme GPU
(non vérifié). Les comparaisons E4/E5 contre E3 utilisent E3 relancé (même session). Moyenne E3 relancé :
0.6609 (écart-type 0.0157).

| Fold | E3 relancé | E4 pondérée | E5 sans augm. |
|---|---:|---:|---:|
| 0 | 0.6554 | 0.6653 | 0.6493 |
| 1 | 0.6472 | 0.6251 | 0.6108 |
| 2 | 0.6469 | 0.6360 | 0.6011 |
| 3 | 0.6802 | 0.6915 | 0.6233 |
| 4 | 0.6748 | 0.6933 | 0.6441 |
| Moyenne | 0.6609 | 0.6622 | 0.6257 |
| Écart-type | 0.0157 | 0.0312 | 0.0208 |

### H3 (E4 contre E3) — confirmée sur la seed 0
| Critère | E3 | E4 | Folds gagnés par E4 |
|---|---:|---:|---:|
| Rappel `akiec` | 0.502 | 0.590 | 4 sur 5 |
| Rappel `df` | 0.437 | 0.670 | 5 sur 5 |
| Rappel `vasc` | 0.743 | 0.852 | 5 sur 5 |
| Précision `nv` | 0.896 | 0.935 | 5 sur 5 |
Règle du protocole (moyenne en hausse, ≥ 4 folds sur 5) remplie pour les quatre critères. La macro-F1 ne change
pas (E4 − E3 = +0.0013, E4 gagne 3 folds sur 5) : le gain porte sur les classes rares. Effectifs de validation par
fold : `df` ≈ 21, `vasc` ≈ 26, `akiec` ≈ 59.

### H4 (E5 contre E3) — confirmée sur la seed 0
Macro-F1 : E3 gagne 5 folds sur 5, écart moyen −0.0353 pour E5 (−0.006, −0.037, −0.046, −0.057, −0.031).
Balanced accuracy : 0.634 (E3) contre 0.581 (E5). Rappel de `df` : 0.437 contre 0.272 ; `vasc` : 0.743 contre 0.641.
Écart val_loss − train_loss à la meilleure époque : 0.423 (E3) contre 0.634 (E5). Sans augmentations, meilleure
époque 3 à 6, perte d'entraînement 0.006 à 0.042 : sur-apprentissage rapide.

## E1 — ResNet-18 figé + régression logistique (H1), 5 folds, Kaggle T4, commit `915cea0`
Commande : `python -m scripts.extract_features --train-val <dermtriage-splits>/train_val.csv --image-dir <interim-256>/256`
(`scripts/extract_features.py` : 512 caractéristiques par image, transformations de validation, sans augmentation,
cache `data/features/resnet18_imagenet.npz` non versionné ; `StandardScaler` + `LogisticRegression`, C = 1,0 fixé,
sans poids de classe). Résultats : `results/resnet18_frozen_logreg/fold{k}_seed0/metrics.json`.

| Fold | E1 macro-F1 | Balanced acc. | Accuracy | B1 logreg couleur | Écart E1 − B1 | E2 tête PyTorch |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.4941 | 0.4868 | 0.7288 | 0.2436 | +0.2505 | 0.5425 |
| 1 | 0.4882 | 0.4841 | 0.7255 | 0.2698 | +0.2184 | 0.4581 |
| 2 | 0.5060 | 0.4916 | 0.7299 | 0.2283 | +0.2777 | 0.5179 |
| 3 | 0.5246 | 0.5110 | 0.7386 | 0.2683 | +0.2564 | 0.5185 |
| 4 | 0.4927 | 0.4843 | 0.7370 | 0.2905 | +0.2023 | 0.4982 |
| Moyenne | 0.5011 | | | 0.2601 | +0.2411 | 0.5070 |
Écart-type de E1 : 0.0147.

**H1 confirmée sur 5 folds sur 5** (règle de la phase 1) : les caractéristiques ImageNet figées battent les
caractéristiques couleur de ≈ 0,24 de macro-F1. E2 (tête entraînée avec augmentations) ne bat E1 que sur 3 folds
sur 5 (+0,006 en moyenne) : l'essentiel du gain vient de la représentation, pas de l'entraînement de la tête.
C n'a pas été optimisé.

## Plan restant (guide, phase 3)
Fait : E1 à E5 (5 folds, seed 0), H1 à H4 évaluées. Reste : E6 optionnelle (réseau entier), 3 graines sur la
configuration retenue (E4 envisagée), `scripts/aggregate.py` → `results/comparison.csv`, `notes/RESULTS.md`.

## Estimation du calcul restant (layer4 mesuré : ≈ 33 s par époque sur T4)
| Étape | Runs | Kaggle T4 | CPU local |
|---|---:|---:|---:|
| 3 graines (graines 1 et 2, 5 folds) | 10 | ≈ 1 h 30 | ≈ 12 h |
| **Total restant sans E6** | ≈ 10 | **≈ 1 h 30** | **≈ 12 h** |
| E6 optionnelle | 1 à 5 | + 20 min à 1 h 15 | + 2 h à 10 h |
