# DermTriage

> Classification expérimentale de lésions cutanées à partir du jeu de données
> HAM10000.

DermTriage est un projet d’apprentissage automatique consacré à l’analyse
d’images dermatologiques. Le dépôt contient actuellement les briques d’un
pipeline PyTorch : chargement des images, transformations, entraînement et
évaluation d’un modèle de classification à sept classes.

Le projet reste expérimental. Il ne fournit pas de diagnostic médical et ne
remplace ni l’avis ni la prise en charge d’un professionnel de santé.

## Fonctionnalités disponibles

- chargement des images HAM10000 depuis leurs deux répertoires d’origine ;
- association d’une image à son label à partir d’un `DataFrame` contenant les
  colonnes `image_id` et `label` ;
- transformations d’entraînement et de validation vers des tenseurs
  `3 × 224 × 224`, avec normalisation ImageNet ;
- boucle d’entraînement PyTorch pour une époque ;
- évaluation avec la perte, l’exactitude, l’exactitude équilibrée et la
  macro-F1 ;
- tests unitaires du jeu de données, des transformations et des fonctions
  d’entraînement.

## Organisation du dépôt

```text
.
├── src/
│   ├── dataset.py       # jeu de données PyTorch
│   ├── train.py         # entraînement et évaluation
│   └── transforms.py    # prétraitements des images
├── tests/               # tests unitaires
├── notes/               # exploration et notes d’expérimentation
├── requirements.txt     # dépendances Python verrouillées
└── Makefile             # création de l’environnement et installation
```

## Installation

Cloner le dépôt, puis créer l’environnement virtuel et installer les
dépendances :

```bash
git clone <URL_DU_DEPOT>
cd DermTriage
make install
```

La commande crée l’environnement `.venv`, met `pip` à jour et installe les
versions définies dans `requirements.txt`.

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
dépôt — puis conserver la structure suivante dans le répertoire de données
choisi :

```text
<data_dir>/
├── HAM10000_images_part_1/
│   └── *.jpg
└── HAM10000_images_part_2/
    └── *.jpg
```

Le tableau transmis à `DermDataset` doit au minimum fournir :

- `image_id` : nom du fichier sans l’extension `.jpg` ;
- `label` : entier compris entre `0` et `6`.

Le mapping entre les diagnostics d’origine et ces indices doit être préparé en
amont et rester identique pour tous les sous-ensembles de données.

## Lancer les tests

Depuis la racine du dépôt et dans l’environnement virtuel :

```bash
python -m pytest
```

Les tests vérifient notamment la forme et la normalisation des images, la
gestion explicite des fichiers absents, la mise à jour des poids pendant
l’entraînement et le calcul des métriques d’évaluation.

## État et prochaines étapes

Les composants de base du pipeline sont présents et couverts par des tests. Le
projet ne propose pas encore d’application utilisateur ni de dispositif de
triage clinique complet.

- [x] Charger et transformer les images HAM10000
- [x] Implémenter les primitives d’entraînement et d’évaluation
- [x] Ajouter les premiers tests unitaires
- [ ] Formaliser le script d’entraînement de bout en bout
- [ ] Versionner les configurations et les résultats d’expériences
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
3. lancez la suite de tests ;
4. ouvrez une pull request décrivant le besoin, l’approche et les limites.

Toute contribution liée à la santé doit rester prudente, sourcée et explicite
sur son périmètre d’utilisation.

## Licence

Ce projet est distribué sous licence [MIT](LICENSE).
