# DermTriage

> Classification expérimentale de lésions cutanées à partir du jeu de données
> HAM10000.

DermTriage est un projet d’apprentissage automatique consacré à l’analyse
d’images dermatologiques. Le dépôt contient un pipeline PyTorch complet
(partitions reproductibles, chargement des images, entraînement et évaluation
d’un ResNet-18 à sept classes), des baselines, six variantes expérimentales
comparées sur 5 folds et les scripts qui régénèrent les tableaux et les figures.

Le projet reste expérimental. Il ne fournit pas de diagnostic médical et ne
remplace ni l’avis ni la prise en charge d’un professionnel de santé.

## Fonctionnalités disponibles

- partitions reproductibles par lésion (`StratifiedGroupKFold`), écrites dans
  `data/processed/` : `splits.csv`, `train_val.csv` et `test.csv` ;
- mapping des classes figé dans `src/classes.py`
  (`akiec, bcc, bkl, df, mel, nv, vasc`), sans `LabelEncoder` ;
- chargement des images HAM10000, depuis les deux répertoires d’origine ou depuis
  des copies pré-redimensionnées à 256 px ;
- transformations d’entraînement et de validation vers des tenseurs
  `3 × 224 × 224`, avec normalisation ImageNet ;
- boucle d’entraînement PyTorch pour une époque (arguments nommés obligatoires)
  et évaluation avec la perte, l’exactitude, l’exactitude équilibrée et la
  macro-F1 ;
- poids de classes robustes à une classe absente ;
- baselines (classifieur naïf et régression logistique sur des features couleur) ;
- script d’entraînement reproductible d’un fold (`scripts/train.py`), configurations
  YAML par variante, arrêt précoce sur la macro-F1 de validation ;
- ResNet-18 figé comme extracteur de caractéristiques (512 dimensions, mises en
  cache) suivi d’une régression logistique (`scripts/extract_features.py`) ;
- figures de la phase 3 reconstruites à partir de `results/` (`scripts/make_figures.py`) ;
- chronométrage d’une époque ResNet-18 ;
- tests unitaires (55), dont des tests sur les partitions réelles.

## Organisation du dépôt

```text
.
├── src/
│   ├── classes.py       # liste figée des classes
│   ├── dataset.py       # jeu de données PyTorch
│   ├── models.py        # ResNet-18 (tête, layer4 ou tout entraînable)
│   ├── train.py         # entraînement, prédiction, évaluation
│   ├── transforms.py    # prétraitements des images
│   └── utils.py         # poids de classes, métriques, sauvegarde
├── scripts/
│   ├── make_splits.py   # partitions : splits.csv, train_val.csv, test.csv
│   ├── preprocess.py    # copie des images avec le petit côté à 256 px
│   ├── baselines.py     # baselines sur les 5 folds
│   ├── train.py         # entraînement d'un fold, écrit runs/<config>/fold<k>_seed<s>/
│   ├── extract_features.py # E1 : ResNet-18 figé + régression logistique, 5 folds
│   ├── make_figures.py  # figures de la phase 3, lues depuis results/
│   └── time_epoch.py    # durée d’une époque ResNet-18
├── configs/             # une configuration YAML par variante (tête, layer4, pondérée, sans augmentations)
├── tests/               # tests unitaires et tests des partitions
├── notebooks/           # exploration, baseline d’origine et prototype d’entraînement
├── notes/               # notes d’exploration, de baseline et protocole expérimental
├── rapport_data/        # informations structurées par phase pour le rapport
├── results/             # métriques, historiques et logits de validation par variante et par fold
├── figures/             # figures de la phase 3 (PNG)
├── runs/                # sorties brutes des entraînements, dont les poids (non versionné)
├── requirements.txt     # dépendances d’exécution
├── requirements-dev.txt # dépendances de développement (pytest, jupyterlab)
└── Makefile             # commandes du projet
```

## Installation

Cloner le dépôt, puis créer l’environnement virtuel et installer les
dépendances :

```bash
git clone <URL_DU_DEPOT>
cd DermTriage
make install
```

La commande crée l’environnement `.venv` et installe `requirements-dev.txt`
(qui inclut `requirements.txt`). Les versions de `torch` ne sont pas figées :
`pip` choisit la variante CPU ou GPU adaptée à la machine.

Pour activer ensuite l’environnement :

```bash
source .venv/bin/activate
```

Pour le supprimer :

```bash
make clean
```

## Préparer les données

Télécharger HAM10000 séparément — les images ne sont pas versionnées dans ce
dépôt (`data/` est ignoré par Git) — et placer dans `data/raw/` :

```text
data/raw/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
│   └── *.jpg
└── HAM10000_images_part_2/
    └── *.jpg
```

Puis générer les partitions :

```bash
make splits
```

Colonnes de `splits.csv`, `train_val.csv` et `test.csv` :
`image_id, lesion_id, dx, label, fold, split`. Le test (`fold = -1`, environ
10 % des images) est isolé dans `test.csv` : pour développer, ne lire que
`train_val.csv` (folds 0 à 4).

Les partitions dépendent des versions de scikit-learn et de numpy : sur une autre
machine, copiez les CSV existants plutôt que de les régénérer, afin d’évaluer
tous les modèles sur exactement les mêmes images.

Pour pré-redimensionner les images (petit côté à 256 px, dans
`data/interim/256/`) :

```bash
make preprocess
```

`DermDataset` accepte `data/raw` ou `data/interim/256` comme répertoire de données.

## Commandes

| Commande | Rôle |
|---|---|
| `make install` | crée `.venv` et installe les dépendances |
| `make splits` | génère les partitions dans `data/processed/` |
| `make preprocess` | copie les images à 256 px dans `data/interim/256/` |
| `make test` | lance les tests |
| `make baselines` | écrit `results/baselines/fold{0..4}.json` (≈ 5 min sur CPU) |
| `make features` | extrait les caractéristiques ResNet-18 figées et écrit `results/resnet18_frozen_logreg/` (≈ 10 min sur CPU, ≈ 2 min sur GPU) |
| `make time-epoch` | mesure la durée d’une époque ResNet-18 sur 20 lots |

## Entraîner un modèle

```bash
python -m scripts.train --config configs/resnet18_head.yaml --fold 0 --seed 0
```

Chaque lancement écrit dans `runs/<nom_config>/fold<k>_seed<s>/` :

| Fichier | Contenu |
|---|---|
| `config.yaml` | configuration exacte et hash du commit Git |
| `history.csv` | époque, pertes, macro-F1 de validation, durée |
| `best.pt` | poids de la meilleure époque (arrêt précoce sur la macro-F1) |
| `val_logits.csv` | `image_id, lesion_id, label, logit_0 … logit_6` |
| `metrics.json` | métriques finales du fold |

Seul `train_val.csv` est lu : le jeu de test n'est jamais chargé. Les logits sont
sauvegardés pour calibrer et évaluer l'abstention sans réentraîner. Un GPU est
conseillé (une époque ≈ 0,6 min sur T4 avec les images à 256 px). L'option `--limit N`
permet un contrôle rapide, dont les résultats ne sont pas valides.

Variantes disponibles (une seule différence entre deux variantes voisines) :

| Config | Variante |
|---|---|
| `resnet18_head.yaml` | tête seule, avec augmentations (E2) |
| `resnet18_layer4.yaml` | `layer4` + tête, lr 1e-4 (E3) |
| `resnet18_layer4_weighted.yaml` | E3 avec perte pondérée par les poids de classes (E4) |
| `resnet18_layer4_noaug.yaml` | E3 sans augmentations (E5) |

Le protocole expérimental (hypothèses, usage des données, règle de décision) est
fixé dans `notes/PROTOCOL.md`. Les figures se régénèrent avec
`python -m scripts.make_figures`.

Les tests des partitions sont ignorés si `data/processed/splits.csv` est absent.

## Résultats

Macro-F1 de validation, moyenne des 5 folds (graine 0) ; un fold = environ 1 800 images,
partitions identiques pour toutes les variantes :

| Variante | Macro-F1 moyenne | Écart-type entre folds |
|---|---:|---:|
| B0 · classe majoritaire | 0,115 | 0,000 |
| B1 · régression logistique sur 30 features couleur | 0,260 | 0,024 |
| E1 · ResNet-18 figé + régression logistique | 0,501 | 0,015 |
| E2 · ResNet-18, tête seule | 0,507 | 0,032 |
| E3 · `layer4` + tête | 0,661 | 0,016 |
| E4 · E3 avec perte pondérée | 0,662 (graine 0) ; 0,654 sur 3 graines | 0,031 |
| E5 · E3 sans augmentations | 0,626 | 0,021 |

Sur la graine 0, avec la règle du protocole (moyenne en hausse et au moins 4 folds sur 5) :

- **H1** (caractéristiques ImageNet contre couleurs) : confirmée, 5 folds sur 5, environ +0,24 de macro-F1 ;
- **H2** (adapter `layer4`) : confirmée, 5 folds sur 5, environ +0,15 contre la tête seule ;
- **H3** (perte pondérée) : confirmée sur les classes rares (rappel de `df`, `vasc`, `akiec`)
  et la précision de `nv`, sans gain de macro-F1 ;
- **H4** (augmentations) : confirmée, 5 folds sur 5, environ +0,035 de macro-F1.

Ces résultats portent sur la graine 0. Pour E4, 3 graines donnent un écart-type de 0,0075 entre graines (0,024 entre folds) ; E3 n’a qu’une graine.
Les exécutions sur GPU Kaggle ne sont pas reproductibles à l’identique d’une session à
l’autre (E3 relancé : −0,001 à −0,005 de macro-F1), contrairement à deux lancements sur la
même machine. Détails, tableaux par fold et commandes : `rapport_data/phase_3_transfert_ablations/`.

Durée d’une époque ResNet-18 (7 210 images) : environ 0,6 min sur un GPU Tesla T4
(Kaggle, images à 256 px ; 1,2 min avec les JPEG d’origine), environ 15 à 18 min sur CPU.

## État et prochaines étapes

Les phases 0 à 2 (assainissement, protocole, entraînement reproductible) sont terminées ;
la phase 3 (transfert et ablations) est en cours. Un tiers peut cloner le dépôt, lancer
`make install && make splits && make test && make baselines` et retrouver les baselines.
Le projet ne propose pas encore d’application utilisateur ni de dispositif de triage
clinique complet.

- [x] Charger et transformer les images HAM10000
- [x] Implémenter les primitives d’entraînement et d’évaluation
- [x] Ajouter les premiers tests unitaires
- [x] Rendre les partitions et le mapping des classes reproductibles
- [x] Établir des baselines de référence
- [x] Mesurer le coût de calcul d’une époque
- [x] Écrire le protocole expérimental (`notes/PROTOCOL.md`)
- [x] Écrire le script d’entraînement reproductible et ses tests
- [x] Valider l’entraînement complet du fold 0 et sa reproductibilité
- [x] Comparer six variantes sur 5 folds (E1 à E5, baselines) et évaluer H1 à H4
- [x] Versionner les configurations, les résultats et les figures de la phase 3
- [ ] Mesurer la variance entre graines (graines 1 et 2 sur la configuration retenue)
- [ ] Agréger les résultats (`results/comparison.csv`) et rédiger `notes/RESULTS.md`
- [ ] Analyser les erreurs, calibrer et étudier l’abstention (phases 4 et 5)
- [ ] Évaluer une seule fois sur le jeu de test (phase 6)
- [ ] Ajouter une interface d’inférence
- [ ] Documenter les performances, les biais et les limites du modèle
- [ ] Réaliser une validation clinique avant tout usage réel

## Limites et usage responsable

HAM10000 est un jeu de données de recherche et ne représente pas nécessairement
toutes les populations, tous les appareils de prise de vue ni toutes les
situations cliniques. Les prédictions d’un modèle entraîné sur ces données
peuvent être erronées ou biaisées.

N’utilisez pas DermTriage pour prendre une décision médicale. En cas de lésion
inquiétante, d’évolution rapide, de douleur importante ou de doute, consultez
un professionnel de santé ou les services d’urgence appropriés.

## Contribution

Les contributions sont les bienvenues. Avant de proposer une modification :

1. créez une branche dédiée ;
2. accompagnez le changement de tests lorsque cela s’applique ;
3. lancez la suite de tests (`make test`) ;
4. ouvrez une pull request décrivant le besoin, l’approche et les limites.

Toute contribution liée à la santé doit rester prudente, sourcée et explicite
sur son périmètre d’utilisation.

## Licence

Ce projet est distribué sous licence [MIT](LICENSE).
