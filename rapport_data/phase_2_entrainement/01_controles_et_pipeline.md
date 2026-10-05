# Phase 2 — Script d'entraînement reproductible

**Commande :** `python -m scripts.train --config configs/resnet18_head.yaml --fold 0 --seed 0`
**Sorties :** `runs/<nom_config>/fold<k>_seed<s>/` (`config.yaml`, `history.csv`,
`best.pt`, `val_logits.csv`, `metrics.json`). `runs/` n'est pas versionné.

## Pipeline
- `src/models.py` : `build_model(name, pretrained, trainable)`, `trainable ∈ {head, layer4, all}`,
  `fc` remplacée par `Linear(512, 7)`, poids ImageNet `IMAGENET1K_V1`.
- `src/train.py` : `predict` renvoie `(logits, labels)` dans l'ordre du dataset ;
  `evaluate` l'appelle et ajoute `logits` et `labels` à son retour.
- `src/utils.py` : `set_seed` (random, numpy, torch, `Generator`) et `seed_worker`.
- `scripts/train.py` : lit `train_val.csv` uniquement (refuse toute ligne `split=test`),
  arrêt précoce sur la macro-F1 de validation, tous les fichiers viennent de la meilleure
  époque, `config.yaml` contient le hash du commit Git.
- `configs/resnet18_head.yaml` (AdamW 1e-3, 20 époques, batch 64, patience 4, sans poids
  de classe, augmentations) et `configs/resnet18_layer4.yaml` (lr 1e-4).
- Les logits sont sauvegardés plutôt que les probabilités : le *temperature scaling*
  divise les logits par T, et l'abstention se calcule sans réentraîner.

## Tests (51 au total)
`predict` (forme et ordre), logits renvoyés par `evaluate`, même graine → mêmes poids après
une époque, `trainable="head"` ne laisse que `fc`, le script écrit les cinq fichiers, deux
lancements identiques → même macro-F1 à 1e-4 près, le test n'est jamais chargé.

## Contrôles de bon sens (Kaggle, Tesla T4, `notebooks/train_prototype.ipynb`)
| Contrôle | Résultat | Attendu |
|---|---|---|
| Perte initiale (512 images mélangées, tête aléatoire sur features ImageNet) | 1,988 | ≈ ln(7) = 1,946 |
| Sur-apprentissage de 32 images (`trainable=all`, sans augmentation, Adam 1e-3) | perte 0,0001 à l'itération 99 (0,0015 à la 10ᵉ) | < 0,05 |
| Une époque, tête seule, fold 0 | train 0,9645, 34,3 s | sans erreur |

- **Perte initiale.** Un modèle qui ne sait rien répartit sa confiance quasi uniformément
  sur 7 classes : la cross-entropy d'une loi uniforme vaut −ln(1/7) = ln(7). La mesure doit
  se faire sur des images mélangées : un lot du fold de validation, trié par lésion, est
  presque mono-classe et donnait 1,31 au lieu de 1,95.
- **Époque de contrôle, fold 0 (non retenue comme résultat) :** accuracy 0,7155, balanced
  accuracy 0,2606, macro-F1 0,2765. Une seule époque, un seul fold, sans arrêt précoce :
  ce chiffre ne sert pas à décider H1, qui se jugera sur les 5 folds selon `PROTOCOL.md`.

## Entraînement complet du fold 0 (Kaggle, Tesla T4, commit `91e8538`)
`configs/resnet18_head.yaml`, seed 0. Fichiers versionnés dans
`results/resnet18_head/fold0_seed0/` (`metrics.json`, `history.csv`, `config.yaml`) ;
`best.pt` et `val_logits.csv` restent hors du dépôt (`runs*/`).

| Élément | Valeur |
|---|---|
| Meilleure époque | 14 (19 époques lancées, arrêt précoce, patience 4) |
| Macro-F1 de validation | **0,5425** (0,5424631950301545) |
| Balanced accuracy | 0,5262 |
| Accuracy | 0,7476 |
| Perte de validation | 0,6980 |
| Durée d'une époque | ≈ 24 à 26 s |

Référence : régression logistique couleur, macro-F1 0,2436 (même fold). Résultat provisoire
sur un seul fold : H1 se décide sur les 5 folds (protocole).

Observations :
- La macro-F1 de validation oscille entre 0,47 et 0,54 après l'époque 3 (0,5425 à
  l'époque 14, 0,508 à l'époque 15) : la « meilleure époque » est en partie un pic de
  bruit, ce qui renforce la limite d'optimisme notée dans le protocole.
- Perte d'entraînement (0,62) et de validation (0,66) proches : pas de sur-apprentissage ;
  le modèle à tête seule est plutôt limité en capacité (sujet de H2, non ajusté d'après ce fold).

## Critère de passage : atteint
- Test des 32 images : réussi.
- Entraînement complet du fold 0 : les cinq fichiers sont produits ; `val_logits.csv`
  couvre exactement les 1 803 images (1 345 lésions) du fold 0 ; `config.yaml` contient le
  hash du commit.
- Reproductibilité : deux lancements identiques (`runs/` et `runs_bis/`) donnent des
  `best.pt`, `config.yaml`, `metrics.json` et `val_logits.csv` **identiques au bit près**
  (même somme MD5, écart maximal des logits = 0,0), y compris sur GPU. Seule la colonne
  des durées de `history.csv` diffère. Critère demandé : 1e-4 sur la macro-F1.

## Suite
- Lancer les 5 folds de `resnet18_head` et `resnet18_layer4` (phase 3) et comparer fold par fold.
