# Phase 1 — Protocole expérimental

**Source de référence :** `notes/PROTOCOL.md` (rédigé le 5 octobre 2026, avant toute
expérience avec ResNet-18). Ce fichier en résume les décisions pour le rapport.

## Décisions figées
- **Question :** apport du transfert d'apprentissage sur HAM10000 (7 classes, lésions
  jamais vues) et effet d'une abstention calibrée sur l'erreur des cas acceptés.
- **Métrique principale :** macro-F1, décidée avant les expériences.
- **Hypothèses :** H1 (ImageNet figé contre couleurs), H2 (fine-tuning de `layer4`),
  H3 (poids de classe), H4 (augmentations), H5 (temperature scaling), H6
  (abstention, courbe risque–couverture).
- **Règle de décision :** moyenne sur les 5 folds supérieure et victoire sur au moins
  4 folds sur 5 (comparaison appariée) ; sinon « pas de différence démontrée ».
- **Usage des données :** entraînement sur 4 folds ; validation pour l'arrêt précoce,
  le choix d'époque et la température T ; prédictions *out-of-fold* pour comparer les
  variantes et choisir le seuil d'abstention ; test (1 002 images) lu une seule fois
  en phase 6.

## Limites écrites dans le protocole
- Le fold de validation sert deux fois (arrêt précoce et température) : scores
  *out-of-fold* légèrement optimistes. Variante stricte possible : couper chaque fold
  de validation en deux moitiés par lésion.
- `df` et `vasc` ont une vingtaine d'images par fold : F1 très variable, macro-F1 à
  interpréter avec prudence.
- Les baselines (F1 0,1146 et 0,2436, fold 0) étaient connues avant la rédaction ;
  aucun résultat de réseau de neurones ne l'était.

## À faire plus tard
- Recalculer les baselines sur les 5 folds pour la comparaison appariée de H1.
- Toute modification du protocole est datée et justifiée dans la section
  « Modifications » de `notes/PROTOCOL.md`.
