# DermTriage — Protocole expérimental

**Date de rédaction :** 5 octobre 2026
**Statut :** figé avant toute expérience avec ResNet-18. Toute modification
ultérieure est consignée dans la section « Modifications » en fin de document,
avec sa date et sa justification.

Ce protocole fixe la question, les hypothèses, l'usage de chaque jeu de données,
les métriques et la règle de décision. Il est écrit avant d'avoir vu le moindre
résultat d'un réseau de neurones : les seuls résultats connus au moment de la
rédaction sont ceux des deux baselines de la phase 0 (voir §7), qui servent de
référence.

---

## 1. Question

Sur HAM10000, à lésions jamais vues, quel est l'apport du transfert
d'apprentissage pour la classification en 7 classes, et une règle d'abstention
calibrée réduit-elle les erreurs parmi les prédictions acceptées ?

Le projet est expérimental. Il ne vise pas un usage clinique.

---

## 2. Données

| Élément | Valeur |
|---|---|
| Jeu de données | HAM10000, 10 015 images, 7 classes |
| Classes (ordre figé, `src/classes.py`) | `akiec, bcc, bkl, df, mel, nv, vasc` |
| Partition | `StratifiedGroupKFold` stratifié par classe, groupé par `lesion_id`, `random_state=42` |
| Test | 1 002 images (747 lésions), `fold = -1` |
| `train_val` | 9 013 images (6 723 lésions), 5 folds de 1 802 à 1 803 images |
| Fichiers | `data/processed/splits.csv`, `train_val.csv`, `test.csv` |

Les partitions sont par lésion : une même lésion n'apparaît jamais à la fois
dans l'entraînement et dans la validation ou le test. Les 5 folds et le test
sont identiques pour tous les modèles et toutes les variantes.

Images par classe dans les partitions (aucune classe absente d'un fold) :

| Classe | test | par fold |
|---|---:|---:|
| akiec | 32 | 59 |
| bcc | 51 | 92–93 |
| bkl | 110 | 197–198 |
| df | 12 | 20–21 |
| mel | 112 | 200–201 |
| nv | 671 | 1 206–1 207 |
| vasc | 14 | 25–26 |

Remarque : `df` et `vasc` n'ont qu'une vingtaine d'images par fold. Leur F1 par
fold variera fortement, et la macro-F1 (moyenne non pondérée des 7 classes)
héritera de cette variabilité. Les écarts faibles entre variantes devront être
interprétés avec prudence.

---

## 3. Hypothèses

| # | Hypothèse | Comparaison | Mesure |
|---|---|---|---|
| H1 | Les caractéristiques ImageNet battent les couleurs | ResNet-18 figé + tête linéaire contre régression logistique sur features couleur | Macro-F1 |
| H2 | Adapter les dernières couches aide | Fine-tuning de `layer4` contre tête seule | Macro-F1 |
| H3 | Pondérer la perte aide les classes rares | Avec contre sans poids de classe | Rappel de `df`, `vasc`, `akiec` ; précision de `nv` |
| H4 | Les augmentations réduisent le sur-apprentissage | Avec contre sans augmentations | Macro-F1, écart entre train et validation |
| H5 | Le *temperature scaling* améliore la confiance sans changer les classes prédites | Avant contre après calibration | Log-loss, score de Brier, ECE |
| H6 | S'abstenir réduit l'erreur sur les cas acceptés | Courbe risque–couverture | Erreur à 80 % de couverture, AURC |

Une variante est toujours comparée à une référence unique, nommée avant
l'expérience : la baseline logistique pour H1, la tête seule pour H2, le
modèle sans poids pour H3, le modèle sans augmentations pour H4, le modèle
non calibré pour H5, l'absence d'abstention pour H6.

---

## 4. Usage de chaque jeu de données

| Données | Images | Usage autorisé | Interdit |
|---|---:|---|---|
| Folds d'entraînement (4 sur 5) | ≈ 7 210 | Apprendre les poids | — |
| Fold de validation (le 5e) | ≈ 1 803 | Arrêt précoce, choix de l'époque, ajustement de la température T | Apprendre les poids |
| Prédictions *out-of-fold* (les 5 folds de validation réunis) | 9 013 | Comparer les variantes, choisir le seuil d'abstention | — |
| Test | 1 002 | Une seule évaluation finale (phase 6) | Choisir quoi que ce soit |

- Le test n'est lu par aucun script avant la phase 6. Les scripts de
  développement lisent `train_val.csv`, jamais `splits.csv`.
- Les hyperparamètres de chaque variante sont fixés avant de lancer les 5 folds
  et ne sont réglés que sur des folds de validation.
- Le seuil d'abstention est choisi sur les prédictions *out-of-fold*, jamais sur
  le test : le choisir sur le test fausserait la courbe risque–couverture que
  l'on veut montrer, puisque le seuil serait ajusté sur les données mêmes qui
  servent à l'évaluer.

**Limite assumée.** Le fold de validation sert à la fois à l'arrêt précoce et à
l'ajustement de la température. Les scores *out-of-fold* sont donc légèrement
optimistes. C'est la pratique standard (Guo et al. ajustent T sur la
validation), et le test reste propre. Une variante plus stricte, si le temps le
permet : couper chaque fold de validation en deux moitiés par lésion (une pour
l'arrêt précoce, une pour T).

---

## 5. Métriques

**Principale (décidée ici, jamais changée) :** macro-F1 sur les 7 classes.

**Secondaires :**
- balanced accuracy ;
- rappel par classe ;
- matrice de confusion ;
- temps d'inférence par image.

L'accuracy brute n'est pas retenue comme critère : la classe `nv` représente
environ 67 % des images, et un classifieur qui prédit toujours `nv` obtient
66,94 % d'accuracy.

**Vue « triage » :** sensibilité du groupe malin (`mel`, `bcc`, `akiec`) contre
le reste. C'est l'erreur qui coûte cher en clinique : un mélanome classé `nv`.

**Calibration (H5) :** log-loss, score de Brier, ECE.
**Abstention (H6) :** courbe risque–couverture, erreur à 80 % de couverture, AURC.

---

## 6. Règle de décision

Une variante « gagne » si les deux conditions suivantes sont réunies :

1. sa macro-F1 moyenne sur les 5 folds est supérieure à celle de la référence ;
2. elle bat la référence sur au moins 4 folds sur 5.

Les folds étant identiques d'une variante à l'autre, la comparaison est
**appariée, fold par fold**.

Si l'une des conditions n'est pas remplie, la conclusion est
« pas de différence démontrée ». C'est un résultat valable, pas un échec.

Pour H3, la décision porte sur le rappel de `df`, `vasc` et `akiec` et sur la
précision de `nv`, avec la même règle (moyenne sur les 5 folds et au moins 4
folds sur 5). Pour H5, la calibration réussit si la log-loss, le score de Brier
et l'ECE baissent, et si les classes prédites sont inchangées.

---

## 7. Références connues avant les expériences

Baselines de la phase 0, fold 0 (validation) contre folds 1 à 4
(entraînement), `results/baselines/fold0.json` :

| Modèle | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Classifieur majoritaire (`nv`) | 0,6694 | 0,1429 | 0,1146 |
| Régression logistique (30 features couleur) | 0,6628 | 0,2282 | 0,2436 |

Ces valeurs portent sur le fold 0 seul. Les baselines seront recalculées sur les
5 folds pour la comparaison appariée de H1.

---

## 8. Reproductibilité

- Mêmes partitions pour tous les modèles (CSV figés, copiés et non régénérés
  d'une machine à l'autre : les partitions dépendent des versions de
  scikit-learn et de numpy).
- Graine aléatoire fixée dans chaque configuration d'entraînement.
- Chaque résultat est écrit dans `results/` avec sa configuration.
- Le calcul d'entraînement se fait sur GPU (Kaggle T4 : environ 1,2 min par
  époque ResNet-18, contre environ 15 à 18 min sur CPU).

---

## Modifications

*(Aucune au 5 octobre 2026. Toute modification sera datée et justifiée ici.)*
