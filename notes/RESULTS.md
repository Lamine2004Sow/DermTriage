# Résultats de la phase 3 (graine 0, 5 folds)

Verdicts des hypothèses H1 à H4 selon la règle de `notes/PROTOCOL.md` (§ 6) : une variante gagne si sa macro-F1
moyenne dépasse celle de la référence **et** si elle la bat sur au moins 4 folds sur 5 (comparaison appariée).
Sources : `results/comparison.csv` (`make aggregate`), détails par fold dans
`rapport_data/phase_3_transfert_ablations/01_resultats_5_folds.md`, figures dans `figures/`
(`python -m scripts.make_figures`). Le jeu de test n'a pas été lu.

## Tableau de synthèse

| ID | Variante | Macro-F1 (moyenne ± écart-type entre folds) | Référence | Écart moyen | Folds gagnés |
|---|---|---:|---|---:|---:|
| B0 | Classe majoritaire | 0,115 ± 0,000 | – | – | – |
| B1 | Logreg sur 30 features couleur | 0,260 ± 0,024 | B0 | +0,146 | 5 |
| E1 | ResNet-18 figé + logreg | 0,501 ± 0,015 | B1 | +0,240 | 5 |
| E2 | ResNet-18, tête seule | 0,507 ± 0,032 | E1 | +0,007 | 3 |
| E3 | `layer4` + tête | 0,661 ± 0,016 | E2 | +0,154 | 5 |
| E4 | E3 + perte pondérée (3 graines) | 0,654 ± 0,024 | E3 | −0,007 | 1 |
| E5 | E3 sans augmentations | 0,626 ± 0,021 | E3 | −0,035 | 0 |

## Verdicts

| Hypothèse | Comparaison | Verdict | Preuve |
|---|---|---|---|
| **H1** Les caractéristiques ImageNet battent les couleurs | E1 contre B1 | **Confirmée** | +0,240 de macro-F1, 5 folds sur 5 |
| **H2** Adapter les dernières couches aide | E3 contre E2 | **Confirmée** | +0,154, 5 folds sur 5 (avec le E3 d'origine : +0,157, 5 sur 5) |
| **H3** Pondérer la perte aide les classes rares | E4 contre E3 | **Confirmée** sur les classes rares, **sans gain de macro-F1** | Rappel `akiec` 0,502 → 0,612, `df` 0,437 → 0,600, `vasc` 0,743 → 0,843 ; précision `nv` 0,896 → 0,934 ; ≥ 4 folds sur 5 pour chacun des quatre critères avec chacune des 3 graines |
| **H4** Les augmentations réduisent le sur-apprentissage | E3 contre E5 | **Confirmée** | +0,035 de macro-F1, 5 folds sur 5 ; écart perte de validation − perte d'entraînement à la meilleure époque 0,42 (avec) contre 0,63 (sans) |
| **H5** Le *temperature scaling* améliore la confiance | – | Non évaluée | phase 5 |
| **H6** S'abstenir réduit l'erreur sur les cas acceptés | – | Non évaluée | phase 5 |

## Ce que les résultats disent

- L'essentiel du gain vient de la **représentation** (B1 → E1, +0,24) puis de l'**adaptation de `layer4`**
  (E2 → E3, +0,15). Entraîner une tête sur des caractéristiques figées (E1 → E2) n'ajoute presque rien
  (+0,007, 3 folds sur 5 : non démontré).
- La **perte pondérée** déplace les erreurs sans améliorer la macro-F1 : elle rend les classes rares mieux
  reconnues (rappel de `df` de 0,44 à 0,60) et fait monter la précision de `nv` (0,90 → 0,93) à macro-F1 égale.
  Le gain de macro-F1 de E4 vu sur la graine 0 (+0,001) disparaît avec les graines 1 et 2 (0,649 et 0,650).
- Sans **augmentations**, le modèle sur-apprend dès l'époque 3 à 6 (perte d'entraînement 0,006 à 0,04) et perd
  surtout sur les classes rares (rappel de `df` 0,27, de `vasc` 0,64 contre 0,44 et 0,74 avec).
- Les courbes d'apprentissage de E3 (`figures/fig5_courbes_perte_E3.png`) montrent la perte de validation qui
  remonte dès l'époque 2 à 4 : l'arrêt précoce (patience 4) retient les époques 4 à 19 selon le fold.
- Les confusions dominantes de E4 (`figures/fig6_confusion_E4.png`) sont `mel → nv` (0,19), `bkl → mel` (0,14)
  et `bkl → nv` (0,12).

## Limites

- **Une seule graine** pour E1, E2, E3 et E5 ; E4 en a trois. Le bruit d'entraînement à fold fixe est d'environ
  0,011 de macro-F1 (E4), l'écart entre graines de 0,0075 sur la moyenne de 5 folds, l'écart entre folds de 0,024.
  Les écarts de H2 (+0,15) et de H1 (+0,24) sont bien supérieurs à ce bruit ; celui de H4 (+0,035) l'est de peu ;
  celui de E1 → E2 (+0,007) ne l'est pas.
- **Reproductibilité entre sessions :** sur Kaggle, E3 relancé donne 0,001 à 0,005 de macro-F1 de moins que
  l'exécution d'origine. Les comparaisons E3/E4/E5 utilisent E3 relancé, de la même session que E4 et E5.
  E3 n'a donc qu'une graine, et la comparaison avec E4 (3 graines) n'est pas symétrique.
- **Petits effectifs :** `df` ≈ 21 images et `vasc` ≈ 26 par fold. Le rappel de `df` de E4 varie de 0,506 à 0,670
  selon la graine (écart-type 0,084).
- **C de la régression logistique de E1** fixé à 1,0, non optimisé.
- **Perte non pondérée mesurée sur la validation :** les poids de classes de E4 changent la valeur de la perte, qui
  n'est pas comparable à celle des autres variantes.
- Les résultats portent sur les folds de validation de HAM10000 (images de la même source) ; ils ne disent rien de la
  généralisation à d'autres appareils, populations ou conditions cliniques. Ce projet ne fournit pas de
  diagnostic médical.

## Suite

Phases 4 et 5 (analyse des erreurs avec intervalles de confiance par bootstrap sur les lésions, calibration H5,
abstention H6), puis phase 6 : lecture unique du jeu de test pour la configuration retenue.
