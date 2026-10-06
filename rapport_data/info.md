# Données du rapport

Ce dossier rassemble, de façon structurée, tout ce qui servira à rédiger le
rapport final du projet DermTriage : données, résultats, expériences, décisions
et chiffres exacts.

## Règles
- Un dossier par phase du projet, un fichier par sujet.
- Chaque résultat indique la commande qui le reproduit et le fichier source.
- Les valeurs sont recopiées telles quelles (précision complète), les
  arrondis viennent ensuite.
- Les décisions sont accompagnées de leur justification.
- Les fichiers de ce dossier décrivent ; le code, les données et les
  résultats bruts restent dans `src/`, `scripts/`, `data/` et `results/`.

## Organisation

| Dossier | Contenu |
|---|---|
| `phase_0_assainissement/` | Splits reproductibles, baselines, changements de code, budget de calcul de la phase 0 |
| `phase_1_protocole/` | Décisions du protocole expérimental (résumé de `notes/PROTOCOL.md`) |
| `phase_2_entrainement/` | Pipeline d'entraînement, tests, contrôles de bon sens, fold 0 et reproductibilité |
| `phase_3_transfert_ablations/` | Résultats sur 5 folds (baselines, tête seule), comparaison appariée, plan et budget restants |

## Résultats de référence

| Modèle | F1 macro (fold 0) | F1 macro (moyenne 5 folds, seed 0) | Source |
|---|---:|---:|---|
| Dummy (classe majoritaire) | 0,1146 | 0,1146 | `results/baselines/fold{0..4}.json` |
| Régression logistique (couleur) | 0,2436 | 0,2601 | `results/baselines/fold{0..4}.json` |
| ResNet-18 tête seule | 0,5425 | 0,5070 | `results/resnet18_head/fold{0..4}_seed0/metrics.json` |
| ResNet-18 `layer4` + tête | 0,6594 | 0,6639 | `results/resnet18_layer4/fold{0..4}_seed0/metrics.json` |

## Budget de calcul

| Machine | Une époque ResNet-18 (7 210 images) |
|---|---:|
| Kaggle, Tesla T4 | 1,2 min (images d'origine), 0,57 min (images 256 px) |
| Portable, CPU (extrapolé) | ≈ 17,6 min (≈ 14,6 min avec images 256 px) |
