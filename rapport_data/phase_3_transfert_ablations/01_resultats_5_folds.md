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

## Plan restant (guide, phase 3)
E1 (`scripts/extract_features.py`), E3 fait (voir ci-dessus), E4 (`configs/resnet18_layer4_weighted.yaml`), E5 (`configs/resnet18_layer4_noaug.yaml`), E6 optionnelle (réseau entier), puis `scripts/aggregate.py` →
`results/comparison.csv` et `notes/RESULTS.md`. Criblage sur le fold 0 (E4, E5), 5 folds pour les
meilleures variantes, 3 graines pour la configuration retenue.

## Estimation du calcul restant (layer4 mesuré : ≈ 33 s par époque sur T4)
| Étape | Runs | Kaggle T4 | CPU local |
|---|---:|---:|---:|
| E1 | 1 passe avant | ≈ 2 min | ≈ 10 min |
| Criblage fold 0 (E4, E5) | 2 | ≈ 20 min | ≈ 2,5 h |
| 5 folds pour E4 et E5 (folds 1 à 4) | 8 | ≈ 1 h 10 | ≈ 10 h |
| 3 graines (graines 1 et 2, 5 folds) | 10 | ≈ 1 h 30 | ≈ 12 h |
| **Total de base** | ≈ 20 | **≈ 3 h 30** | **≈ 30 h** |
| E6 optionnelle | 1 à 5 | + 20 min à 1 h 15 | + 2 h à 10 h |
