# DermTriage — Baselines de classification

**Notebook source :** `notebooks/baseline.ipynb`  
**Date :** 26 septembre 2026  
**Auteur :** Lamine SOW  

---

## 1. Objectif

Établir des scores de référence avant l'entraînement d'un réseau de neurones.
Ces baselines permettent de vérifier que le futur modèle ResNet-18 apprend des
informations utiles et fait mieux qu'une prédiction naïve ou qu'un modèle fondé
uniquement sur la couleur.

---

## 2. Protocole expérimental

- **Données :** jeu `train_val` de HAM10000, soit 9 013 images.
- **Entraînement :** folds 1 à 4, soit 7 210 images.
- **Validation :** fold 0, soit 1 803 images.
- **Encodage des classes :** `LabelEncoder` ajusté sur l'ensemble `train_val`.
- **Graine aléatoire :** `random_state=42` pour la régression logistique.
- **Jeu de test :** non utilisé ; il reste réservé à l'évaluation finale.

Le fold 0 n'a pas de propriété particulière. Il est utilisé comme fold de
validation par convention afin d'obtenir une expérience simple et
reproductible. Les folds ayant été construits par lésion, une même lésion ne se
retrouve pas à la fois dans l'entraînement et dans la validation.

### Métriques suivies

| Métrique | Rôle |
|---|---|
| Accuracy | Proportion totale de prédictions correctes |
| Balanced accuracy | Moyenne du rappel obtenu pour chaque classe |
| F1 macro | Moyenne non pondérée du score F1 des sept classes |

La balanced accuracy et le F1 macro sont essentiels ici, car la classe `nv`
représente environ 67 % des images. L'accuracy seule peut donc donner une vision
trompeuse des performances.

---

## 3. Baseline 1 — `DummyClassifier`

Le `DummyClassifier` utilise la stratégie `most_frequent` : il prédit toujours
la classe majoritaire, `nv`. Il constitue le score plancher que tout modèle
utile doit dépasser.

Résultats sur le fold 0 :

- accuracy : **66,94 %** ;
- balanced accuracy : **14,29 %**, soit exactement `1 / 7` ;
- F1 macro : **11,46 %**.

L'accuracy paraît élevée uniquement à cause du déséquilibre des classes. Les
deux métriques macro montrent que ce modèle ne sait pas reconnaître les classes
minoritaires.

---

## 4. Baseline 2 — Features couleur et régression logistique

Chaque image est représentée par **30 variables** calculées sur ses pixels RGB :

- 3 moyennes, une par canal ;
- 3 écarts-types, un par canal ;
- 24 valeurs d'histogramme, soit 8 intervalles normalisés par canal.

Les variables sont standardisées avec `StandardScaler`, puis utilisées par une
`LogisticRegression` configurée avec `max_iter=1000` et
`random_state=42`.

Résultats sur le fold 0 :

- accuracy : **66,28 %** ;
- balanced accuracy : **22,82 %** ;
- F1 macro : **24,36 %**.

Cette baseline obtient une accuracy légèrement inférieure à celle du modèle
naïf, mais elle améliore fortement les métriques macro. Les informations de
couleur permettent donc déjà de distinguer une partie des classes minoritaires,
sans toutefois fournir une classification suffisamment robuste.

---

## 5. Tableau de résultats

| Modèle | Accuracy | Balanced accuracy | F1 macro |
|---|---:|---:|---:|
| `DummyClassifier` | 66,94 % | 14,29 % | 11,46 % |
| `LogisticRegression` (features couleur) | 66,28 % | 22,82 % | 24,36 % |
| ResNet-18 (TP 5) | À venir | À venir | À venir |

---

## 6. Conclusion

Le `DummyClassifier` confirme que l'accuracy brute est peu informative sur ce
jeu très déséquilibré. La régression logistique exploite un signal réel dans les
couleurs : par rapport au modèle naïf, elle gagne **8,53 points** de balanced
accuracy et **12,90 points** de F1 macro, malgré une baisse de **0,67 point**
d'accuracy.

Le ResNet-18 devra donc être comparé en priorité à la balanced accuracy de
**22,82 %** et au F1 macro de **24,36 %**. Ses résultats seront ajoutés au
tableau après l'entraînement du TP 5.
