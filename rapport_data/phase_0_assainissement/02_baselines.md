# Phase 0 — Baselines

**Commande :** `make baselines` → `scripts/baselines.py`
**Sortie :** `results/baselines/fold0.json`
**Entrée :** `data/processed/splits.csv` (lecture directe des labels, plus de `LabelEncoder`)

## Protocole
- Entraînement : folds 1 à 4 = 7 210 images ; validation : fold 0 = 1 803 images.
- Le test n'est pas utilisé.
- Images lues en pleine résolution (`data/raw`), pas dans `data/interim/256`.
- Temps d'exécution observé : environ 10 minutes (extraction des features, CPU).
- Métriques : accuracy, balanced accuracy, F1 macro (`src.utils.compute_metrics`).

## Modèles
1. `DummyClassifier(strategy="most_frequent")` : prédit toujours `nv`.
2. 30 features couleur par image (3 moyennes, 3 écarts-types, 3×8 bins
   d'histogramme normalisés) → `StandardScaler` → `LogisticRegression(max_iter=1000, random_state=42)`.

## Résultats (fold 0)
| Modèle | Accuracy | Balanced accuracy | F1 macro |
|---|---:|---:|---:|
| Dummy | 0,66944 | 0,14286 | **0,11457** |
| LogReg couleur | 0,66278 | 0,22821 | **0,24355** |

Valeurs exactes du JSON : f1 dummy = 0,1145704793545325 ; f1 logreg = 0,2435538079279549.

## Critère de passage de la phase 0
Attendu : f1 = 0,1146 (dummy) et 0,2436 (logreg). **Obtenu : identique.**

## À retenir
- La logreg gagne +8,53 pts de balanced accuracy et +12,90 pts de F1 macro sur le
  dummy, pour −0,67 pt d'accuracy : la couleur porte déjà un signal.
- Ces deux valeurs sont la référence à battre pour ResNet-18, sur le même fold 0
  et les mêmes images.
