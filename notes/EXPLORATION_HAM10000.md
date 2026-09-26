# DermTriage — Suivi de l'exploration des données

**Notebook source :** `exploration.ipynb`  
**Date :** 26 septembre 2026  
**Auteur :**  

---

## 1. Ce qui a été fait

- Chargement des métadonnées HAM10000 (`HAM10000_metadata.csv`)
- Exploration des images des deux lots (`part_1` et `part_2`)
- Vérification de la cohérence fichiers ↔ CSV (identifiants, doublons)
- Analyse de la structure lésion / image (plusieurs images par lésion)
- Séparation des données sans fuite par lésion (`StratifiedGroupKFold`)
- Mapping des chemins d'images
- Encodage des labels
- Sauvegarde de `splits.csv`

---

## 2. Structure des données

| Colonne | Description |
|---|---|
| `image_id` | Identifiant unique de l'image |
| `lesion_id` | Identifiant de la lésion (plusieurs images possibles) |
| `dx` | Classe diagnostique (7 catégories) |
| `fold` | Numéro de fold (0 à 4) ou -1 pour le test |
| `split` | `train_val` ou `test` |

---

## 3. Répartition des données

| Split | Images | Lésions |
|---|---|---|
| train_val | 9 013 | 6 723 |
| test | 1 002 | 747 |
| **total** | **10 015** | **7 470** |

### Répartition par fold (train_val uniquement)

| Fold | Images | Lésions |
|---|---|---|
| 0 | 1 803 | 1 345 |
| 1 | 1 803 | 1 345 |
| 2 | 1 803 | 1 345 |
| 3 | 1 802 | 1 344 |
| 4 | 1 802 | 1 344 |

### Distribution des classes

| Classe | Total | % | train_val | % | test | % |
|---|---|---|---|---|---|---|
| nv | 6 705 | 66.9% | 6 034 | 66.9% | 671 | 67.0% |
| mel | 1 113 | 11.1% | 1 001 | 11.1% | 112 | 11.2% |
| bkl | 1 099 | 11.0% | 989 | 11.0% | 110 | 11.0% |
| bcc | 514 | 5.1% | 463 | 5.1% | 51 | 5.1% |
| akiec | 327 | 3.3% | 295 | 3.3% | 32 | 3.2% |
| vasc | 142 | 1.4% | 128 | 1.4% | 14 | 1.4% |
| df | 115 | 1.1% | 103 | 1.1% | 12 | 1.2% |

---

## 4. Choix techniques

- **Séparation par lésion** : une même lésion ne peut pas être dans train et val
  simultanément → évite la fuite de données
- **StratifiedGroupKFold** : préserve la distribution des classes dans chaque fold
  malgré le groupement par lésion
- **Graine fixée** : `random_state=42` → résultats reproductibles
- **fold = -1** : valeur sentinelle pour les images réservées au test final
- **Vérification fuite** : 0 lésion en commun entre train_val et test 

---

## 5. Fichiers produits

| Fichier | Emplacement | Description |
|---|---|---|
| `splits.csv` | `data/processed/` | Partitions de toutes les images |

---

## 6. Points d'attention pour la suite

- Dataset **très déséquilibré** : `nv` représente ~67% des images →
  penser à pondérer la perte au TP 6
- `vasc` (1.4%) et `df` (1.1%) sont très rares → métriques par classe
  importantes, pas seulement l'accuracy globale
- Le test est **intouchable** jusqu'au TP 7
- `classe_vers_numero` doit être calculé sur `metadata` entier, pas sur `train` seul