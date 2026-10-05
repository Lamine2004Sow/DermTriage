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

## Reste à faire (critère de passage)
- Entraînement complet du fold 0 avec `scripts/train.py` (cinq fichiers produits).
- Relancer à l'identique : même macro-F1 à 1e-4 près (CPU ; un GPU peut différer
  légèrement malgré la graine, à cause d'opérations non déterministes).
