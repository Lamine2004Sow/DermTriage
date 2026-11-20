# Couteau suisse — DermTriage (phase 0 → phase 3)

Pour chaque outil : **l'idée en une phrase**, **un schéma**, **pourquoi ça marche**, **le code**, **les pièges**.
Le code vient de `src/` (bibliothèque du projet) et de `scripts/` (une commande par étape).
Sources : le sujet (`guide_new_version.md`), le protocole (`notes/PROTOCOL.md`), les notes de phase
(`rapport_data/`), la documentation scikit-learn et PyTorch citée dans le guide.
Les chiffres viennent des fichiers de `results/` ou ont été **mesurés en relançant le code** (marqués « mesuré »).
Ce qui est **déduit** (raisonnement, pas mesure) est écrit comme tel.

---

## 0. Quel outil pour quel problème ? (à lire en premier)

| Je veux... | Outil | Section | Lien direct |
|---|---|---|---|
| Donner à chaque classe un **numéro qui ne bouge jamais** | liste figée `CLASSES` | 1 | [→ Bases](#s1) |
| Couper les données **sans fuite** entre entraînement et validation | `StratifiedGroupKFold` par lésion | 2 | [→ Partitions](#s2) |
| Mettre le **test de côté** pour ne jamais le toucher par accident | fichiers séparés + `assert` | 2 | [→ Partitions](#s2) |
| Rendre la lecture des images **plus rapide** | pré-redimensionnement à 256 px | 3 | [→ Préparer les images](#s3) |
| Retrouver une image à partir de son **image_id** | `DermDataset` | 3 | [→ Préparer les images](#s3) |
| Mettre une image au **format attendu par ResNet** | `Resize` + `Normalize` ImageNet | 4 | [→ Transformations](#s4) |
| **Augmenter** les données sans en collecter | flips + `ColorJitter` | 4 | [→ Transformations](#s4) |
| **Mesurer juste** quand une classe fait 67 % des images | macro-F1, balanced accuracy | 5 | [→ Métriques](#s5) |
| Savoir **à partir de quel score** un modèle sert à quelque chose | baselines dummy + couleur | 6 | [→ Baselines](#s6) |
| Choisir **quelles couches apprennent** (tête, `layer4`, tout) | `build_model(trainable=...)` | 7 | [→ Geler / dégeler](#s7) |
| Utiliser un réseau **sans l'entraîner** (sans GPU) | cache de caractéristiques + logreg | 8 | [→ Réseau figé](#s8) |
| Entraîner **une époque** et récupérer les **logits** | `train_one_epoch`, `predict`, `evaluate` | 9 | [→ Boucle](#s9) |
| Faire compter davantage les **classes rares** | `get_class_weights` | 10 | [→ Pondération](#s10) |
| Obtenir **les mêmes chiffres** en relançant | `set_seed`, `seed_worker`, hash Git | 11 | [→ Reproductibilité](#s11) |
| Savoir **quand arrêter** et garder la bonne époque | arrêt précoce sur la macro-F1 | 12 | [→ Arrêt précoce](#s12) |
| Reconnaître le **sur-apprentissage** sur une courbe | écart perte train / perte val | 12 | [→ Arrêt précoce](#s12) |
| Vérifier que le pipeline **n'est pas cassé** avant de lancer | perte initiale ln 7, sur-apprendre 32 images | 13 | [→ Contrôles](#s13) |
| Dire si une variante **bat vraiment** l'autre | comparaison appariée, règle 4 folds sur 5 | 14 | [→ Comparer](#s14) |
| Séparer le bruit **des données** et le bruit **de l'entraînement** | écart-type entre folds / entre graines | 14 | [→ Comparer](#s14) |
| Voir **quelles classes** le modèle confond | matrice de confusion normalisée, rappel | 15 | [→ Erreurs par classe](#s15) |
| Savoir **combien coûte** une époque avant de lancer | `time_epoch.py` + `LimitedLoader` | 16 | [→ Chronométrer](#s16) |
| **Tout refaire** de HAM10000 à `comparison.csv` | chaîne complète | 17 | [→ Chaîne complète](#s17) |

Pas encore codé (pas de section, à ajouter quand le code existera) : bootstrap par lésion (phase 4),
*temperature scaling* et abstention (phase 5, H5 et H6), lecture du test (phase 6), variante E6 (réseau entier).

---

<a id="s1"></a>
## 1. Les bases : données, classes, fichiers (`src/classes.py`, `data/`)

### Les données HAM10000
- **10 015 images** dermatoscopiques (600 × 450 px, JPEG), **7 470 lésions** : une lésion peut avoir jusqu'à
  **6 photos** (1 956 lésions ont plus d'une image, mesuré sur `splits.csv`).
- Une ligne de métadonnées = `image_id` (le fichier), `lesion_id` (la lésion photographiée), `dx` (le diagnostic).
- Classes très déséquilibrées (train_val, mesuré) :

```
nv     6034  ████████████████████████████████████████  67 %
mel    1001  ██████▋                                    11 %
bkl     989  ██████▌                                    11 %
bcc     463  ███                                         5 %
akiec   295  ██                                          3 %
vasc    128  ▊                                           1,4 %
df      103  ▋                                           1,1 %
```
- `df` et `vasc` n'ont qu'une **vingtaine d'images par fold** : leurs scores par fold varient beaucoup.

### Une image vue par le réseau
```
JPEG 600×450 ─▶ PIL RGB ─▶ tenseur (3, 224, 224) float, normalisé ─▶ lot (64, 3, 224, 224) ─▶ logits (64, 7)
```
Un **logit** est un score brut par classe (avant softmax). La classe prédite est `argmax` des logits.

### Une liste de classes figée
```python
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]   # src/classes.py
NUM_CLASSES = len(CLASSES)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
```

### Pourquoi ne pas recalculer le mapping depuis les données ?
Parce qu'un sous-ensemble sans une classe **décale tous les labels**, sans message d'erreur.
Mesuré : sans `df`, `sorted(unique())` donne

```
avec df :  akiec 0  bcc 1  bkl 2  df 3  mel 4  nv 5  vasc 6
sans df :  akiec 0  bcc 1  bkl 2        mel 3  nv 4  vasc 5    ← mel prend le label de df, nv celui de mel...
```
Le modèle apprend alors des étiquettes fausses et les métriques restent plausibles. C'est le « bug silencieux »
du diagnostic du guide (deux notebooks recalculaient le mapping, identiques « par hasard »).

### Où sont les fichiers
| Dossier | Contenu | Versionné ? |
|---|---|---|
| `data/raw/` | images et métadonnées d'origine (1,3 Go + 1,4 Go) | non |
| `data/interim/256/` | images au petit côté 256 px (257 Mo) | non |
| `data/processed/` | `splits.csv`, `train_val.csv`, `test.csv` | non (MD5 notés) |
| `data/features/` | cache `resnet18_imagenet.npz` (9 013 × 512) | non |
| `runs/` | sorties complètes d'un lancement (avec `best.pt`) | non |
| `results/` | `metrics.json`, `history.csv`, `val_logits.csv`, `comparison.csv` | oui |

---

<a id="s2"></a>
## 2. Couper les données sans fuite : partitions par lésion (phase 0 — `scripts/make_splits.py`)

**Idée :** on découpe par **lésion** et non par image, pour qu'une même lésion ne soit jamais à la fois apprise
et évaluée, tout en gardant les mêmes proportions de classes partout.

```
10 015 images (7 470 lésions)
        │  StratifiedGroupKFold(n_splits=10), on garde 1 morceau sur 10
        ├──────────────▶ test : 1 002 images, 747 lésions, fold = -1      (lu une seule fois, phase 6)
        │
        └─▶ train_val : 9 013 images, 6 723 lésions
                 │  StratifiedGroupKFold(n_splits=5)
                 ▼
            fold 0 │ fold 1 │ fold 2 │ fold 3 │ fold 4        1 802 ou 1 803 images chacun
            ───────┼────────┼────────┼────────┼───────
   run k :  valider sur le fold k, apprendre sur les 4 autres

Lésion HAM_0001601 (2 photos) :  ■ ■  ──▶ toutes les deux dans le fold 0 (mesuré)
```
- **Groupé** (`groups=lesion_id`) : toutes les photos d'une lésion vont ensemble.
- **Stratifié** (`y=dx`) : chaque fold a ≈ 67 % de `nv`, ≈ 20 `df`, etc.

### Pourquoi grouper par lésion ?
Deux photos de la même lésion se ressemblent presque comme deux copies. Si l'une est dans l'entraînement et
l'autre dans la validation, le modèle est noté sur une image qu'il a **presque déjà vue** : le score mesure la
mémoire, pas la généralisation à **une lésion jamais vue** (la question du protocole).
Mesuré : avec un `train_test_split` aléatoire à 20 %, **704 lésions** sont des deux côtés et **754 images de
validation sur 2 003 (38 %)** ont une sœur dans l'entraînement.

### Pourquoi stratifier en plus ?
Avec 103 `df` en tout, un découpage seulement groupé peut laisser un fold presque sans `df`. Sa F1 serait alors
indéfinie ou très bruitée, et la macro-F1 du fold aussi. Le protocole vérifie qu'**aucune classe n'est absente
d'un fold** (20 à 21 `df`, 25 à 26 `vasc` par fold).

### Pourquoi deux niveaux (test, puis 5 folds) ?
Chaque partition a un seul rôle (protocole §4) : les folds servent à choisir (arrêt précoce, comparaison des
variantes), le test sert à **une seule** mesure finale. Un test utilisé pour choisir n'est plus un test.

```python
test_splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
rest_idx, test_idx = next(                                       # 1 morceau sur 10 = ~10 % pour le test
    test_splitter.split(metadata, y=metadata["dx"], groups=metadata["lesion_id"])
)
rest = metadata.iloc[rest_idx].copy()
test = metadata.iloc[test_idx].copy()

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
rest["fold"] = -1
for fold, (_, val_idx) in enumerate(cv.split(rest, y=rest["dx"], groups=rest["lesion_id"])):
    rest.iloc[val_idx, rest.columns.get_loc("fold")] = fold     # chaque image reçoit le fold où elle valide
...
splits["label"] = splits["dx"].map(CLASS_TO_INDEX).astype(int)  # mapping figé (section 1)
```
```python
assert set(splits.loc[splits["split"] == "train_val", "lesion_id"]).isdisjoint(   # preuve : aucune lésion
    splits.loc[splits["split"] == "test", "lesion_id"]                             # commune train_val / test
)
splits[splits["split"] == "train_val"].to_csv(TRAIN_VAL_OUTPUT, index=False)       # les scripts de dev ne
splits[splits["split"] == "test"].to_csv(TEST_OUTPUT, index=False)                 # lisent que celui-ci
```

### Les astuces du code
- **`next(test_splitter.split(...))`** : un `KFold` à 10 morceaux dont on ne prend que le premier, c'est un
  découpage 90/10 qui garde à la fois le groupement et la stratification (ce que `train_test_split` ne fait pas).
- **Une colonne `fold`** au lieu de 5 fichiers : un seul CSV décrit les 5 découpages ; le run `k` filtre
  `fold != k` / `fold == k`.
- **`train_val.csv` séparé** : tous les scripts de développement le lisent et vérifient
  `(train_val["split"] == "train_val").all()`. Le test ne peut pas être chargé par erreur.
- **`.iloc[val_idx, get_loc("fold")]`** : `split()` renvoie des **positions**, pas des index pandas.

### Ce qu'on observe (mesuré sur `splits.csv`)
- test : 1 002 images, 747 lésions ; train_val : 9 013 images, 6 723 lésions ;
  folds 0 à 2 : 1 803 images, folds 3 et 4 : 1 802.

### Pièges
- **Régénérer les partitions sur une autre machine** : sur Kaggle, mêmes métadonnées (même MD5) mais partitions
  **différentes** (`2f975107…` contre `132c1875…`), à cause de la version de scikit-learn ou numpy.
  → On **copie** les CSV, on ne les régénère pas ; MD5 de référence dans
  `rapport_data/phase_0_assainissement/04_calcul.md` ([section Rendu](#s18)).
- **Lire `splits.csv` dans un script de dev** : il contient le test. → Toujours `train_val.csv`.

Source : scikit-learn, *Cross-validation: evaluating estimator performance*, section « Cross-validation iterators
for grouped data », https://scikit-learn.org/stable/modules/cross_validation.html (cité dans le guide).

---

<a id="s3"></a>
## 3. Préparer les images : pré-redimensionner et retrouver un fichier (phase 0 — `scripts/preprocess.py`, `src/dataset.py`)

**Idée :** on réduit **une fois pour toutes** chaque image à un petit côté de 256 px, pour ne plus décoder des JPEG
600 × 450 à chaque époque ; le `Dataset` retrouve ensuite chaque fichier par son `image_id`.

```
data/raw/HAM10000_images_part_1/ISIC_0024306.jpg  600×450  ─┐
                                                            ├─ resize_short_side(256), BICUBIC ─▶ 341×256
data/raw/HAM10000_images_part_2/...                        ─┘   data/interim/256/ISIC_0024306.jpg

   ratio 4:3 conservé : 600/450 = 341/256 ≈ 1,33          1,3 Go + 1,4 Go ──▶ 257 Mo
```

### Pourquoi 256 et pas directement 224 ?
Les transformations font ensuite `Resize((224, 224))` (section 4). Partir de 256 laisse une marge si on change un
jour pour `Resize(224)` + `CenterCrop(224)` ou un recadrage aléatoire (déduit ; c'est la taille classique avant un
crop 224 dans les tutoriels ImageNet).

### Pourquoi ça accélère ?
Décoder un JPEG coûte à peu près en proportion de son nombre de pixels (déduit) : 341 × 256 = 87 k pixels contre 270 k.
Mesuré (`04_calcul.md`) : sur CPU le gain est faible (**−17 %** par lot, 4,68 → 3,87 s), le calcul du réseau domine ;
sur GPU T4 le décodage devient le goulot, l'époque passe de **74 s à 34 s** (montages pas strictement identiques).

```python
def resize_short_side(image, short_side=SHORT_SIDE):
    width, height = image.size
    scale = short_side / min(width, height)            # le PETIT côté devient 256
    return image.resize(
        (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC
    )
...
    if target.exists():
        continue                                       # relançable : on saute ce qui est fait
```
```python
# src/dataset.py — un dictionnaire image_id -> chemin, construit une fois
dossiers = [self.data_dir / "HAM10000_images_part_1",
            self.data_dir / "HAM10000_images_part_2",
            self.data_dir]                             # data/raw (2 sous-dossiers) OU data/interim/256 (plat)
self.chemins = {f.stem: f for dossier in dossiers if dossier.is_dir() for f in dossier.glob("*.jpg")}
...
image = Image.open(image_path).convert("RGB")
```

### Les astuces du code
- **`self.df.reset_index(drop=True)`** : après un filtre `fold != k`, l'index pandas a des trous ; `iloc[idx]` avec
  un index propre garantit que l'élément `idx` du loader est la ligne `idx` du DataFrame (indispensable pour
  aligner `val_logits.csv` sur `val_df`, [section 9](#s9)).
- **Le dictionnaire `chemins`** est construit une fois dans `__init__` : pas de recherche de fichier à chaque image.
- **`.convert("RGB")`** : garantit 3 canaux même si un fichier était en niveaux de gris ou en RGBA.
- **`quality=95`** à l'écriture : limite les artefacts d'une deuxième compression JPEG.

### Pièges
- **`FileNotFoundError: Image introuvable pour image_id : ...`** : mauvais `--image-dir`, ou pré-traitement pas fait.
- `scripts/baselines.py` lit `data/raw` (pleine résolution) et non `data/interim/256` : ses features couleur ne sont
  pas calculées sur les mêmes pixels que le réseau. Sans effet sur les conclusions, mais à savoir.

---

<a id="s4"></a>
## 4. Mettre l'image au format ResNet et l'augmenter (phase 2 — `src/transforms.py`)

**Idée :** on présente au réseau des images **comme celles d'ImageNet** (224 × 224, mêmes moyenne et écart-type), et
pendant l'entraînement on les **retourne** et on **varie la lumière** au hasard pour qu'il n'apprenne pas par cœur.

```
Entraînement : 341×256 ─▶ Resize(224,224) ─▶ flip H ─▶ flip V ─▶ ColorJitter ─▶ ToTensor ─▶ Normalize
Validation :   341×256 ─▶ Resize(224,224) ─────────────────────────────────▶ ToTensor ─▶ Normalize
                                            (p = 0,5)  (p = 0,5)

Normalize, canal R :  pixel 0,60  ──▶  (0,60 − 0,485) / 0,229 = 0,50
```

### Pourquoi les moyennes `[0.485, 0.456, 0.406]` ?
Ce sont les statistiques d'ImageNet, sur lesquelles les poids du ResNet ont été appris. Le réseau pré-entraîné
« attend » des entrées dans cette échelle ; d'autres statistiques déplaceraient toutes les activations de la
première couche.

### Pourquoi des flips horizontaux **et** verticaux ?
Une lésion vue au dermatoscope n'a ni haut ni bas : la retourner donne une image tout aussi plausible, avec le même
diagnostic. (Sur des photos de chats, le flip vertical serait une mauvaise idée.) Déduit du sujet ; mesuré par H4 :
sans augmentations (E5), macro-F1 **0,626 contre 0,661**, et le rappel de `df` tombe de 0,44 à 0,27.

### Pourquoi aucune augmentation en validation ?
On veut **une seule** prédiction, la même à chaque relance, sur l'image telle qu'elle est. Une augmentation
aléatoire rendrait le score de validation bruité et non reproductible.

```python
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),                      # écrase le 4:3 (choix à justifier, voir Pièges)
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),   # éclairage et appareil varient
        transforms.ToTensor(),                              # uint8 [0,255] -> float [0,1], (H,W,C) -> (C,H,W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
```

### Pièges
- **`Resize((224, 224))` écrase le ratio 4:3** (341 × 256 → 224 × 224) : la lésion est un peu aplatie. Ce n'est pas
  une erreur (train et val sont traités pareil), c'est un choix. Alternative : `Resize(224)` + `CenterCrop(224)`,
  qui garde les proportions mais coupe les bords. Le guide : ne changer ça **qu'en expérience à part**.
- **`ColorJitter` sur la teinte** n'est pas utilisé : la couleur porte une partie du diagnostic (déduit ; la
  baseline couleur seule atteint déjà 0,26 de macro-F1 contre 0,11 pour le dummy).
- La config `augment: false` remplace les transformations d'entraînement par celles de validation
  (`scripts/train.py` : `get_train_transforms() if cfg.get("augment", True) else get_val_transforms()`).

---

<a id="s5"></a>
## 5. Mesurer juste quand une classe domine : macro-F1, balanced accuracy (`src/utils.py::compute_metrics`)

**Idée :** on donne **le même poids à chaque classe**, pour qu'un modèle qui ne reconnaît que `nv` ait un mauvais
score même s'il a raison 67 % du temps.

```
Modèle « toujours nv » sur le fold 0 (1 803 images, mesuré)

             rappel   précision   F1
akiec         0         –         0
bcc           0         –         0
bkl           0         –         0
df            0         –         0
mel           0         –         0
nv            1,00     0,669     0,802     F1 = 2·P·R / (P + R) = 2·0,669·1 / 1,669
vasc          0         –         0
                                 ─────
accuracy           = 1 207 / 1 803 = 0,669   ← a l'air correct
balanced accuracy  = moyenne des rappels = 1/7 = 0,143
macro-F1           = moyenne des F1 = 0,802 / 7 = 0,115   ← dit la vérité
```

### Pourquoi la macro-F1 et pas l'accuracy ?
L'accuracy est une moyenne **sur les images**, donc dominée par `nv`. La macro-F1 est une moyenne **sur les
classes** : rater tous les `df` (1 % des images) coûte 1/7 du score. C'est la métrique principale, fixée avant les
expériences (protocole §5).

### Pourquoi la F1 et pas seulement le rappel ?
Le rappel seul se gagne en prédisant souvent une classe rare ; la F1 combine **rappel** (« je trouve les `df` ») et
**précision** (« quand je dis `df`, c'est vrai »). La balanced accuracy, elle, ne regarde que les rappels.

### Sa faiblesse ici
Avec ≈ 20 `df` par fold, **un seul `df` de plus ou de moins** bien classé bouge le rappel de `df` de 0,05, donc la
macro-F1 de ≈ 0,007 (déduit : 0,05 / 7, sans compter l'effet sur la précision). C'est l'ordre de grandeur de l'écart
E1 → E2 (+0,007), d'où le verdict « non démontré » ([section 14](#s14)).

```python
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)   # classe jamais prédite : F1 = 0
    }
```

### Les astuces du code
- **`zero_division=0`** : si une classe n'est jamais prédite, sa précision vaut 0/0 ; on la met à 0 (sans
  avertissement) au lieu de planter ou d'afficher un warning à chaque époque.
- La fonction prend des **classes prédites** (`argmax`), pas des logits : les appelants font l'`argmax`.

### Pièges
- **`average="weighted"`** au lieu de `"macro"` repondère par l'effectif : on retombe sur un score dominé par `nv`.

Source : scikit-learn, `f1_score` et `balanced_accuracy_score`, https://scikit-learn.org/stable/modules/model_evaluation.html.

---

<a id="s6"></a>
## 6. Poser un plancher : baselines dummy et couleur (phase 0 — `scripts/baselines.py`)

**Idée :** avant tout réseau, on mesure ce que donnent **rien** (toujours la classe majoritaire) et **la couleur
seule** ; un modèle qui ne bat pas ces deux scores n'apprend rien d'utile.

```
image ─▶ pixels / 255 ─▶ 3 moyennes + 3 écarts-types + 3 histogrammes de 8 cases = 30 nombres
                                                                │
                                         StandardScaler ─▶ LogisticRegression ─▶ classe

Histogramme du canal R (8 cases sur [0, 1]) :  ▁▁▂▃▅█▆▂   (fraction des pixels par case)

Macro-F1 moyenne sur 5 folds (results/comparison.csv) :
B0 dummy        0,115  ██▎
B1 couleur      0,260  █████▏
E1 ResNet figé  0,501  ██████████
```

### Pourquoi le dummy ?
Il donne le **plancher** exact de chaque métrique (accuracy 0,669, macro-F1 0,115) : on sait ainsi qu'une accuracy
de 0,70 n'a rien d'impressionnant.

### Pourquoi des features couleur ?
C'est l'hypothèse simple à battre : « la couleur suffit-elle ? ». Elle donne +0,146 sur le dummy (5 folds sur 5) mais
reste loin d'un réseau : la **forme et la texture** de la lésion comptent (H1, [section 8](#s8)).

### Pourquoi un `StandardScaler` ?
Les moyennes sont dans [0, 1], les fractions d'histogramme souvent < 0,1 : sans mise à l'échelle, la régularisation
L2 de la régression logistique pénaliserait les features de façon inégale et l'optimiseur convergerait mal.

```python
def extract_features(image_path):
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    pixels = pixels.reshape(-1, 3)                                     # (H·W, 3) : un pixel par ligne
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, channel], bins=8, range=(0.0, 1.0))
        histograms.append(hist / pixels.shape[0])                      # fractions : indépendant de la taille
    features = np.concatenate([means, stds, *histograms]).astype(np.float32)
    assert features.shape == (30,)
    return features
...
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(np.zeros((len(y_train), 1)), y_train)                         # X ignoré : seul y compte
```

### Les astuces du code
- **`np.zeros((n, 1))`** pour le dummy : il ignore `X`, mais scikit-learn exige un tableau de la bonne longueur.
- **Les features sont calculées une fois** (`all_features`) puis indexées par `is_val` pour chaque fold.
- **`range=(0.0, 1.0)`** fixe les bornes des cases : sans ça, chaque image aurait ses propres cases et les
  histogrammes ne seraient pas comparables.

### Ce qu'on observe (`results/baselines/`)
- Dummy : macro-F1 0,115 sur les 5 folds (écart-type 0,00004). Couleur : 0,260 ± 0,024 ; fold 0 : 0,2436.

---

<a id="s7"></a>
## 7. Choisir quelles couches apprennent : tête, `layer4` ou tout (phase 2 — `src/models.py`)

**Idée :** on part d'un ResNet-18 **déjà entraîné sur ImageNet**, on remplace sa dernière couche (1 000 classes) par
une couche à 7 sorties, et on décide quelles couches ont le droit de changer.

```
image ─▶ conv1 ─▶ layer1 ─▶ layer2 ─▶ layer3 ─▶ layer4 ─▶ avgpool ─▶ fc 512→7 ─▶ logits
         └──────────── bords, textures, motifs ─────┘    └ formes ┘   (512 nombres)

trainable="head"    ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄    🔥 fc         3 591 paramètres
trainable="layer4"  ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ ❄ 🔥 layer4 🔥           🔥 fc     8 397 319 paramètres
trainable="all"     🔥 tout le réseau                                                   11 180 103 paramètres
(❄ = requires_grad False, 🔥 = appris)
```
Nombres de paramètres **calculés à la main** à partir de l'architecture (fc : 512 × 7 + 7 = 3 591 ;
`layer4` : 8 393 728 ; total avec fc à 7 sorties : 11 689 512 − 513 000 + 3 591), non mesurés ici (torchvision
absent de l'environnement local).

### Pourquoi des poids ImageNet aident sur de la peau ?
Les premières couches d'un CNN apprennent des motifs **génériques** (bords, coins, textures, taches de couleur), utiles
pour n'importe quelle image. Seules les dernières sont spécifiques aux classes d'ImageNet. C'est le principe du
**transfert d'apprentissage** (tutoriel PyTorch cité dans le guide). Mesuré : un ResNet **figé** (E1) atteint 0,501
contre 0,260 pour la couleur.

### Pourquoi `layer4` et pas tout le réseau ?
`layer4` est le bloc le plus **spécifique** à ImageNet : c'est lui qu'il faut adapter à la dermatologie. Le dégeler
fait passer la macro-F1 de **0,507 à 0,661** (H2, 5 folds sur 5). Tout dégeler (E6, non fait) coûte plus cher et risque
d'abîmer les couches génériques avec seulement 7 200 images d'entraînement.

```python
def build_model(name="resnet18", pretrained=True, trainable="head"):
    ...
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)   # 512 -> 7, poids aléatoires

    for param in model.parameters():
        param.requires_grad = trainable == "all"            # tout gelé (ou tout libre si "all")
    if trainable == "layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True                          # la tête apprend toujours
    return model
```
```python
# scripts/train.py — l'optimiseur ne reçoit que ce qui apprend
params = [p for p in model.parameters() if p.requires_grad]
```

### Les astuces du code
- **Remplacer `fc` avant de geler** : la nouvelle couche est créée avec `requires_grad=True`, puis la boucle remet
  tout à `False`, puis on la rallume. L'ordre « tout éteindre, rallumer ce qu'on veut » évite d'oublier une couche.
- **`model.fc.in_features`** plutôt que `512` en dur : marche aussi pour un autre ResNet.
- **Filtrer les paramètres dans l'optimiseur** : PyTorch saute déjà les paramètres sans gradient, mais la liste
  courte rend l'intention claire et l'optimiseur ne garde d'état que pour ce qui apprend.

### Tête seule ou `layer4` ?
| | tête seule (E2) | `layer4` + tête (E3) |
|---|---|---|
| Paramètres appris | 3 591 | 8,4 M |
| Learning rate (config) | 1e-3 | 1e-4 |
| Macro-F1 (5 folds) | 0,507 ± 0,032 | **0,661 ± 0,016** |
| Perte train à la fin (fold 0) | ≈ 0,62, proche de val | 0,11, val remonte |
| Problème | **capacité** limitée | **sur-apprentissage** → arrêt précoce |

### Pièges
- ⚠️ **BatchNorm des blocs gelés en mode train.** `requires_grad=False` gèle les **poids**, pas les **statistiques**
  de BatchNorm (`running_mean`, `running_var`) : elles sont mises à jour à chaque passe avant en mode `model.train()`.
  Or `train_one_epoch` appelle `model.train()` sur tout le modèle. Donc en E2 et E3, les BatchNorm de conv1 à layer3
  **continuent de s'adapter** aux images HAM10000 (déduit de la doc PyTorch de `BatchNorm2d` ; non testé dans le
  dépôt). Le guide demande de les laisser en `eval()` et de le vérifier par un test. Correction possible :
  après `model.train()`, remettre en `eval()` les modules `nn.BatchNorm2d` dont les paramètres sont gelés.
- **Un seul learning rate** pour `layer4` et la tête (1e-4) : le guide suggère un lr 10 fois plus faible pour
  `layer4` que pour la tête (groupes de paramètres de l'optimiseur). Pas fait.
- **`ValueError: trainable doit être dans ('head', 'layer4', 'all')`** : faute de frappe dans la config.

Source : PyTorch, *Transfer learning for computer vision tutorial*,
https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html ; doc `torch.nn.BatchNorm2d`.

---

<a id="s8"></a>
## 8. Utiliser un réseau sans l'entraîner : cache de caractéristiques + régression logistique (E1 — `scripts/extract_features.py`)

**Idée :** si le réseau est **figé** et l'image **pas augmentée**, le vecteur de 512 nombres de chaque image est
toujours le même ; on le calcule **une fois**, puis on entraîne une simple régression logistique dessus en
quelques secondes, sans GPU.

```
                 une seule passe avant (≈ 9 013 images)                   pour chaque fold k (secondes)
images ─▶ ResNet-18 figé, fc = Identity ─▶ (9 013, 512) ─▶ cache .npz ─▶ StandardScaler ─▶ LogReg(C=1)
                                                │                        apprend sur folds ≠ k
                                                └── relu + avgpool : valeurs ≥ 0, moyenne 0,61 (mesuré)
                                                                    valide sur fold k ─▶ decision_function
                                                                                               = 7 « logits »
```

### Pourquoi c'est possible ?
Un réseau figé en mode `eval()` est une **fonction fixe** : même entrée, même sortie. Sans augmentation, l'entrée de
chaque image est toujours la même, donc sa sortie aussi. Les 5 folds réutilisent le même tableau.

### Pourquoi E1 n'a pas d'augmentations alors qu'E2 en a ? (question du sujet)
Avec des augmentations, chaque passage d'une image donnerait un vecteur **différent** : il faudrait repasser toutes
les images dans le réseau à chaque époque, et le cache n'aurait plus de sens. E2 entraîne la tête en PyTorch,
époque par époque : il peut se le permettre.

### Pourquoi `fc = nn.Identity()` ?
`Identity` renvoie son entrée telle quelle : le modèle s'arrête juste après `avgpool` et renvoie le vecteur de 512
dimensions au lieu des 7 logits.

```python
def build_encoder(pretrained=True):
    model = build_model("resnet18", pretrained=pretrained, trainable="head")
    model.fc = nn.Identity()                        # sortie = le vecteur 512-d de avgpool
    return model.eval()                             # BatchNorm en mode inférence : sortie déterministe

@torch.no_grad()                                    # pas de graphe de calcul : moins de mémoire, plus rapide
def extract(encoder, loader, device):
    encoder.to(device).eval()
    features = [encoder(images.to(device)).cpu() for images, _ in loader]
    return torch.cat(features).numpy().astype(np.float32)
```
```python
if cache.is_file() and not force:
    saved = np.load(cache, allow_pickle=False)
    if np.array_equal(saved["image_id"], ids):      # mêmes images, dans le même ordre ?
        return saved["features"]
    print("Cache incompatible avec train_val.csv : recalcul")
...
scores = model.decision_function(features[is_val])  # (n_val, 7) : scores avant softmax, comme des logits
```

### Les astuces du code
- **Le cache stocke les `image_id`** à côté des features : si `train_val.csv` change (autre ordre, autres images), le
  cache est détecté comme incompatible au lieu de donner des features décalées en silence.
- **`shuffle=False`** dans le `DataLoader` : la ligne `i` des features est l'image `i` de `train_val.csv`.
- **`decision_function`** et non `predict_proba` : même format que les logits des runs PyTorch, donc `aggregate.py`
  et la future calibration lisent les deux pareil.
- **`assert ... isdisjoint(...)`** à chaque fold : preuve, run par run, qu'aucune lésion n'est des deux côtés.
- **`C=1.0` fixé, non optimisé** : choisir `C` sur le fold de validation puis rapporter le score de ce même fold
  serait optimiste. Limite assumée dans `notes/RESULTS.md`.

### Ce qu'on observe
- Cache : (9 013, 512), `float32`, minimum 0,0, moyenne 0,61 (mesuré).
- Macro-F1 0,501 ± 0,015, **+0,240 sur la couleur, 5 folds sur 5 → H1 confirmée**.

### Réseau figé + logreg (E1) ou tête PyTorch (E2) ?
| | E1 | E2 |
|---|---|---|
| Modèle de la tête | linéaire (logreg L2, solveur exact) | linéaire (`fc`, AdamW, 20 époques max) |
| Augmentations | non | oui |
| Coût | 1 passe avant + secondes | 20 époques × 5 folds sur GPU |
| Macro-F1 | 0,501 ± 0,015 | 0,507 ± 0,032 (+0,007, 3 folds sur 5 : **non démontré**) |
Les deux têtes sont des modèles **linéaires** sur le même genre de caractéristiques : il est logique qu'elles se
valent (déduit). Le gain arrive quand on adapte `layer4`.

---

<a id="s9"></a>
## 9. Entraîner une époque et récupérer les logits (phase 2 — `src/train.py`)

**Idée :** une époque = passer **une fois** sur toutes les images d'entraînement, lot par lot, en corrigeant les poids
après chaque lot ; l'évaluation repasse sur la validation **sans gradient** et garde les **logits**.

```
pour chaque lot (64 images) :
   zero_grad ─▶ outputs = model(images) ─▶ loss = CE(outputs, labels) ─▶ backward ─▶ optimizer.step
                (64, 7)                     (un nombre)                  gradients    poids ← poids − lr·…

perte de l'époque = Σ (perte du lot × taille du lot) / nombre d'images
   lots : 64, 64, ..., 64, 42 (dernier lot incomplet)   ← chaque IMAGE compte pareil
```

### Pourquoi `loss.item() * len(labels)` et pas une moyenne des pertes de lots ?
`CrossEntropyLoss` renvoie la **moyenne** sur le lot. Si le dernier lot n'a que 42 images, moyenner les moyennes
donnerait à ces 42 images le même poids que 64 autres. Multiplier par la taille du lot puis diviser par le nombre
total d'images donne la **vraie moyenne par image** (testé dans `tests/test_train.py`, cité par le guide).

### Pourquoi `model.eval()` change les sorties d'un ResNet ? (question du sujet)
À cause de **BatchNorm** : en mode train, elle normalise avec la moyenne et la variance **du lot courant** (la sortie
d'une image dépend des autres images du lot) ; en mode eval, elle utilise les **statistiques mémorisées**
(`running_mean`, `running_var`). ResNet-18 n'a pas de dropout ; c'est donc la seule couche en cause.

### Pourquoi garder les logits et pas les probabilités ? (question du sujet)
Le *temperature scaling* (phase 5) divise les **logits** par T avant le softmax, et l'abstention se calcule à partir
des scores : avec les logits sauvegardés, ces deux phases **ne demandent aucun réentraînement**. Les probabilités se
recalculent à tout moment avec un softmax ; l'inverse perd de l'information numérique pour les très petites probas.

```python
def train_one_epoch(model, loader, *, optimizer, criterion, device):   # le * oblige à nommer ces arguments
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()                       # sinon les gradients s'additionnent d'un lot à l'autre
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)     # .item() : un float Python, détaché du graphe
    return total_loss / len(loader.dataset)

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(images.to(device)).cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)   # dans l'ordre du dataset (loader sans mélange)
```

### Les astuces du code
- **`*` dans la signature** : `train_one_epoch(model, loader, opt, crit)` lève une `TypeError`. L'ancienne version
  acceptait un appel positionnel où optimiseur et perte pouvaient être **inversés en silence** (diagnostic du guide).
- **`.cpu()` dans `predict`** : on libère la mémoire GPU au fur et à mesure et on peut passer aux outils numpy.
- **`evaluate` calcule la perte sur tous les logits d'un coup** : c'est la moyenne exacte sur le fold.

### Pièges
- **Oublier `optimizer.zero_grad()`** : les gradients s'accumulent, l'entraînement diverge.
- **`val_loss` de E4 non comparable** : avec `class_weights: true`, le même `criterion` pondéré sert en validation ;
  sa perte n'est pas comparable à celle des variantes sans poids (limite notée dans `RESULTS.md`).
- **Oublier `torch.no_grad()`** en évaluation : résultat juste mais mémoire qui explose (le graphe est gardé).

---

<a id="s10"></a>
## 10. Faire compter davantage les classes rares : perte pondérée (H3 — `src/utils.py::get_class_weights`)

**Idée :** dans la perte, une erreur sur une image `df` (82 images d'entraînement) **coûte 59 fois plus** qu'une erreur
sur une image `nv` (4 827) ; le modèle cesse d'ignorer les classes rares.

```
poids_c = N / (K · n_c)        N = 7 210 images, K = 7 classes, n_c = effectif de la classe c (fold 0, mesuré)

classe   n_c    poids
akiec    236    4,36   ████▎
bcc      371    2,78   ██▊
bkl      791    1,30   █▎
df        82   12,56   ████████████▌
mel      801    1,29   █▎
nv     4 827    0,21   ▏
vasc     102   10,10   ██████████

Perte pondérée (PyTorch, reduction="mean") :  L = Σ_i w_{y_i} · CE_i  /  Σ_i w_{y_i}
```
- Avec ces poids, **chaque classe pèse autant au total** : `n_c × poids_c = N / K` pour toutes (= 1 030).

### Que fait la pondération, mathématiquement ? (question d'entretien)
Elle multiplie la perte de chaque image par le poids de **sa vraie classe**. Le gradient venant des images rares est
amplifié : le modèle déplace sa frontière de décision **vers** `nv`, il prédit plus souvent les classes rares.
C'est la formule `class_weight="balanced"` de scikit-learn.

### Quel compromis ? (question du sujet)
Prédire plus souvent `df`, `mel`... signifie prédire **moins souvent `nv`** : le rappel de `nv` baisse, sa précision
monte. Mesuré (E3 → E4, seed 0, 5 folds cumulés) :

| | E3 sans poids | E4 avec poids |
|---|---:|---:|
| Rappel `df` | 0,44 | **0,67** |
| Rappel `akiec` | 0,50 | 0,59 |
| Rappel `vasc` | 0,74 | 0,85 |
| `mel` classé `nv` | 0,35 | **0,19** |
| Rappel `nv` | 0,93 | 0,85 |
| Images prédites `nv` (sur 9 013) | 6 256 | 5 468 |
| Accuracy | 0,814 | 0,782 |
| Macro-F1 (5 folds, moyenne sur 3 graines pour E4) | 0,661 | 0,654 |
Les erreurs sont **déplacées**, pas supprimées : H3 confirmée sur les classes rares, **sans gain de macro-F1**.
Pour le triage, c'est le bon sens du déplacement : un mélanome classé `nv` est l'erreur qui coûte cher (protocole §5).

```python
def get_class_weights(df, num_classes=7):
    counts = df["label"].value_counts().reindex(range(num_classes), fill_value=0)   # 7 cases toujours
    total = len(df)
    weights = (total / (num_classes * counts.where(counts > 0))).fillna(0.0)        # effectif 0 -> poids 0
    return torch.tensor(weights.values, dtype=torch.float32)
```
```python
# scripts/train.py — poids calculés sur les folds d'ENTRAÎNEMENT du run
weight = get_class_weights(train_df, NUM_CLASSES).to(device) if cfg["loss"]["class_weights"] else None
criterion = nn.CrossEntropyLoss(weight=weight)
```

### Les astuces du code
- **`.reindex(range(num_classes), fill_value=0)`** : `value_counts()` ne liste que les classes **présentes**, triées par
  effectif. Sans `reindex`, une classe absente donnait **6 poids au lieu de 7**, et l'ordre ne suivait pas les labels.
  Correctif écrit en TDD (`test_class_weights_classe_absente`).
- **`counts.where(counts > 0)`** remplace les 0 par `NaN` : la division donne `NaN` au lieu de `inf`, puis
  `fillna(0.0)`. Une classe sans exemple n'est jamais la vraie classe d'une image : son poids n'est jamais utilisé.
- **`.to(device)`** : le tenseur de poids doit être sur le même appareil que les logits.
- **Calculés sur `train_df`**, jamais sur la validation : sinon on utiliserait une information du fold évalué.

### Pièges
- **`RuntimeError` de `CrossEntropyLoss`** sur un sous-échantillon où une classe manque (ancienne version sans
  `reindex`) : poids de taille 6 pour 7 logits.
- **Le rappel de `df` varie de 0,506 à 0,670 selon la graine** (E4, écart-type 0,084) : avec ≈ 20 images, une
  conclusion sur une seule graine est fragile.

Source : scikit-learn, `sklearn.utils.class_weight.compute_class_weight` (« balanced » : `n_samples / (n_classes *
np.bincount(y))`) ; PyTorch, `torch.nn.CrossEntropyLoss` (argument `weight`).

---

<a id="s11"></a>
## 11. Obtenir les mêmes chiffres en relançant : graines et traçabilité (phase 2 — `src/utils.py`, `scripts/train.py`)

**Idée :** on fixe **toutes** les sources de hasard (Python, numpy, torch, ordre de mélange, workers) et on note dans
chaque résultat **quel code** l'a produit.

```
set_seed(0) ─┬─▶ random.seed      ─▶ ...
             ├─▶ np.random.seed   ─▶ ...
             ├─▶ torch.manual_seed ─▶ init de fc, dropout éventuel
             ├─▶ torch.cuda.manual_seed_all
             └─▶ Generator(0) ─▶ DataLoader(shuffle=True) ─▶ ordre des images identique à chaque relance

DataLoader, 2 workers :  worker 0 ─▶ seed_worker ─▶ graine dérivée ─▶ flips et ColorJitter identiques
                         worker 1 ─▶ seed_worker ─▶ autre graine dérivée

config.yaml : git_commit: 64020f6...-dirty   ← « -dirty » = des fichiers suivis étaient modifiés au lancement
```

### Pourquoi un `Generator` en plus de `torch.manual_seed` ?
L'ordre de mélange du `DataLoader` dépend du générateur qu'on lui passe. Lui donner un générateur à soi isole cet
ordre : un appel aléatoire ajouté ailleurs (une init de couche de plus) ne change pas l'ordre des lots.

### Pourquoi `seed_worker` ?
Chaque worker est un **processus à part** avec sa propre copie de `random` et `numpy`. PyTorch donne à chaque worker
une graine torch dérivée du générateur ; `seed_worker` recopie cette graine dans `random` et `numpy`, pour que les
augmentations soient reproductibles quel que soit le nombre de workers.

### Pourquoi un GPU peut-il quand même donner des résultats différents ? (question du sujet)
Certaines opérations cuDNN choisissent leur algorithme ou additionnent dans un ordre non fixe ; l'addition de
flottants n'étant pas associative, le résultat change au dernier bit, puis l'écart grandit au fil des époques.
Mesuré : deux lancements **dans la même session** Kaggle sont identiques au bit près ; **d'une session à l'autre**,
E3 relancé perd 0,001 à 0,005 de macro-F1 (matériel ou versions différentes). D'où la règle : comparer des runs de la
même session.

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator                                  # à passer au DataLoader d'entraînement

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32        # numpy n'accepte que des graines < 2^32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```
```python
# scripts/train.py
dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], ...).stdout.strip()
return out.stdout.strip() + ("-dirty" if dirty else "")
...
full_cfg = {**cfg, "fold": fold, "seed": seed, "git_commit": git_commit(),
            "image_dir": str(image_dir), "limit": limit}
```

### Les astuces du code
- **`--untracked-files=no`** : un fichier non suivi (un notebook de brouillon) ne rend pas le run « dirty » ;
  seule une modification du code versionné compte.
- **`config.yaml` réécrit avec fold, graine, commit, dossier d'images et `limit`** : le dossier de résultats suffit
  pour savoir comment le refaire.
- **`--limit`** (contrôle rapide) est noté dans la config : un run de contrôle ne peut pas passer pour un vrai.

---

<a id="s12"></a>
## 12. Savoir quand arrêter et lire une courbe d'apprentissage (phase 3 — `scripts/train.py::run`)

**Idée :** après chaque époque on mesure la macro-F1 de validation ; on garde **la meilleure époque** et on s'arrête
quand elle ne s'améliore plus pendant 4 époques (patience).

```
E3, fold 0 (results/resnet18_layer4_rerun/fold0_seed0/history.csv)

époque   perte train   perte val   F1 val    patience
  0        0,820        0,592      0,581       4   ★ meilleure
  1        0,505        0,559      0,615       4   ★
  2        0,409        0,530      0,638       4   ★   ← perte val minimale
  3        0,348        0,548      0,640       4   ★
  4        0,278        0,565      0,620       3
  5        0,236        0,589      0,633       2
  6        0,197        0,598      0,655       4   ★   ← meilleure époque retenue
  7        0,170        0,642      0,644       3
  8        0,133        0,680      0,628       2
  9        0,112        0,655      0,620       1
 10        0,113        0,679      0,634       0   → arrêt précoce

  perte
  0,8 │●                          ● train
      │ ●  ○                      ○ val
  0,6 │○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○    ← val remonte à partir de l'époque 2
      │   ●  ●
  0,4 │       ●  ●
      │            ●  ●
  0,2 │                 ●  ●
      │                      ●  ●   ← train continue de baisser : SUR-APPRENTISSAGE
      └──────────────────────────── époque
```

### À quoi reconnaît-on le sur-apprentissage ? (question du sujet)
La perte d'**entraînement** continue de baisser alors que la perte de **validation** remonte : le modèle apprend des
détails propres aux images d'entraînement. Sans augmentations (E5, fold 0, mesuré), c'est plus brutal : perte train
**0,042 à l'époque 3**, 0,005 à l'époque 7, perte val de 0,575 à 0,683.

### Pourquoi s'arrêter sur la macro-F1 et pas sur la perte de validation ?
La macro-F1 est la métrique de décision (protocole). Ici la perte val est minimale à l'époque 2, mais la F1 continue
de monter jusqu'à l'époque 6 : le modèle devient **trop confiant** (la perte, sensible aux erreurs sûres d'elles,
monte) tout en classant **mieux** (la F1, qui ne regarde que l'`argmax`, monte). Le *temperature scaling* de la phase 5
corrige justement cette confiance excessive (déduit, lien avec Guo et al. 2017 cité dans le guide).

### Pourquoi la patience ?
La F1 de validation est **bruitée** (≈ 20 `df` par fold) : une baisse d'une époque n'est pas une tendance. On tolère
4 époques sans amélioration avant de conclure.

```python
history, best = [], None
best_f1, patience_left = -1.0, cfg["train"]["patience"]
for epoch in range(cfg["train"]["epochs"]):
    train_loss = train_one_epoch(model, train_loader, optimizer=optimizer, criterion=criterion, device=device)
    metrics = evaluate(model, val_loader, criterion, device)
    ...
    if metrics["f1"] > best_f1:
        best_f1, patience_left = metrics["f1"], cfg["train"]["patience"]   # on recharge la patience
        best = {"epoch": epoch, **metrics}                                 # logits de CETTE époque gardés
        save_model(model, out_dir / "best.pt", {...})
    else:
        patience_left -= 1
        if patience_left <= 0:
            print("arrêt précoce")
            break
...
logits = pd.DataFrame(best["logits"].numpy(), columns=[f"logit_{i}" for i in range(NUM_CLASSES)])
logits.insert(0, "label", best["labels"].numpy())
logits.insert(0, "lesion_id", val_df["lesion_id"].values)               # aligné car loader sans mélange
logits.insert(0, "image_id", val_df["image_id"].values)
```

### Les astuces du code
- **`best_f1 = -1.0`** : la première époque est toujours « meilleure », même avec une F1 de 0.
- **`best` garde les logits de la meilleure époque** : `val_logits.csv` et `metrics.json` viennent de la même
  époque que `best.pt`. Écrire les logits de la **dernière** époque serait une incohérence silencieuse.
- **`>` strict** : à égalité, on garde l'époque la plus ancienne (moins d'entraînement).
- **`.values`** à l'insertion : évite un alignement pandas sur l'index (qui produirait des `NaN` si les index
  différaient).

### Ce qu'on observe (meilleure époque, F1 par fold, seed 0)
- E2 (tête) : époques 14, 5, 13, 14, 6 — la F1 oscille entre 0,47 et 0,54 après l'époque 3 : la « meilleure
  époque » est en partie un **pic de bruit**.
- E3 (`layer4`) : époques 6, 5, 19, 4, 7.

### Pièges
- **Le fold de validation sert deux fois** (choisir l'époque, puis noter) : le score out-of-fold est **un peu
  optimiste**. Limite assumée (protocole §4) ; le test reste propre.

---

<a id="s13"></a>
## 13. Vérifier que le pipeline n'est pas cassé : perte initiale et sur-apprentissage de 32 images (phase 2)

**Idée :** deux tests de quelques minutes qui détectent les bugs les plus courants (labels décalés, mauvaise
normalisation, lr absurde) **avant** de payer des heures de GPU.

```
Test 1 — perte initiale                      Test 2 — sur-apprendre 32 images (trainable=all, sans augm.)

modèle qui ne sait rien                      perte
  p = 1/7 pour chaque classe                 2,0 │●
  CE = −ln(1/7) = ln 7 = 1,946               1,0 │ ●
                                             0,1 │  ●●                          0,0015 à l'itération 10
mesuré : 1,988  ✓                            0,0 │     ●●●●●●●●●●●●●●●●●●●●     0,0001 à l'itération 99  ✓
                                                 └────────────────────────── itération
```

### Pourquoi ln(7) ? (question du sujet)
Une tête aléatoire donne des logits petits et proches, donc des probabilités ≈ uniformes, 1/7 par classe. La
cross-entropy d'une image vaut −ln(p de la vraie classe) = −ln(1/7) = ln 7 ≈ 1,95, quelle que soit la classe. Une
perte initiale de 5 ou de 0,3 signale un problème (logits énormes, labels constants...).

### Pourquoi sur-apprendre 32 images ?
Un réseau de 11 M de paramètres **doit** pouvoir apprendre par cœur 32 images. S'il n'y arrive pas, ce n'est pas un
manque de données : c'est un bug (labels mélangés avec les images, gradients coupés, lr à 0). « Le test le plus
rentable en deep learning » (guide).

### Pièges
- **Mesurer la perte initiale sur un lot trié** : un lot du fold de validation, trié par lésion, est presque
  mono-classe ; la perte valait **1,31 au lieu de 1,95**. → Mesurer sur des images **mélangées**
  (`rapport_data/phase_2_entrainement/01_controles_et_pipeline.md`).

---

<a id="s14"></a>
## 14. Dire si une variante bat vraiment l'autre : comparaison appariée (phase 3 — `scripts/aggregate.py`)

**Idée :** comme toutes les variantes utilisent **les mêmes 5 folds**, on compare fold par fold ; une variante
« gagne » si sa moyenne est meilleure **et** si elle gagne sur au moins 4 folds sur 5.

```
            fold 0   fold 1   fold 2   fold 3   fold 4   moyenne
E2 tête     0,542    0,458    0,518    0,519    0,498    0,507
E3 layer4   0,655    0,647    0,647    0,680    0,675    0,661
écart       +0,11    +0,19    +0,13    +0,16    +0,18    +0,154    5 folds sur 5 → GAGNE (H2 confirmée)

Plan : chaque ligne ne change QU'UNE chose par rapport à sa référence
B0 ─▶ B1 ─▶ E1 ─▶ E2 ─▶ E3 ─┬─▶ E4 (poids de classes)
                            └─▶ E5 (sans augmentations)
```

### Pourquoi apparier ?
Certains folds sont **plus durs** que d'autres pour tous les modèles (fold 1 pour E2). L'écart-type entre folds
(0,016 à 0,032) mélange cette difficulté avec l'effet du modèle. La différence **sur le même fold** retire la
difficulté commune : on compare des modèles, pas des folds.

### Pourquoi « 4 sur 5 » en plus de la moyenne ?
Une moyenne peut être tirée par un seul fold très favorable. Exiger 4 folds sur 5 vérifie que le gain est
**régulier**. Si une condition manque : « pas de différence démontrée », un résultat valable (protocole §6).

### Bruit des données ou bruit de l'entraînement ?
Avec 3 graines (E4), on sépare deux sources de variance :
| Source | Mesure | Valeur (E4) |
|---|---|---:|
| Entre folds (quelles images en validation) | écart-type des 5 F1, moyennées sur les graines | 0,024 |
| Entre graines (init, ordre des lots, augmentations) | écart-type des 3 moyennes de 5 folds | 0,0075 |
Un écart entre variantes plus petit que ≈ 0,01 n'est pas lisible avec une seule graine : c'est le cas de E1 → E2
(+0,007) et du +0,001 de E4 sur la seed 0, qui disparaît avec les graines 1 et 2.

```python
def seeds_available(variant_dir):
    """Graines pour lesquelles les 5 folds existent."""
    seeds = None
    for k in FOLDS:
        found = {int(m.group(1)) for p in variant_dir.glob(f"fold{k}_seed*")
                 if (m := re.fullmatch(rf"fold{k}_seed(\d+)", p.name)) and (p / "metrics.json").is_file()}
        seeds = found if seeds is None else seeds & found        # intersection : graines complètes seulement
    return sorted(seeds)
...
by_fold = metrics.mean(axis=0)                 # (graines, folds, métriques) -> moyenne sur les graines
f1 = by_fold[:, 0]
...
diff = f1 - per_fold_f1[ref]                   # écart APPARIÉ, fold par fold
row.update({"ecart_moyen_f1": diff.mean(), "folds_gagnes": int((diff > 0).sum()),
            "gagne_regle_phase_1": bool(diff.mean() > 0 and (diff > 0).sum() >= 4)})
```

### Les astuces du code
- **Le plan est une liste `PLAN`** (id, description, dossier, référence, changement) : la règle « un seul
  changement » est écrite dans le code, et `comparison.csv` dit pour chaque ligne contre quoi elle est comparée.
- **Intersection des graines** : une graine à qui il manque un fold est ignorée au lieu de fausser la moyenne.
- **Moyenne sur les graines avant l'écart** : la comparaison reste appariée par fold même quand E4 a 3 graines et
  E3 une seule.
- **`ddof=1`** : écart-type d'échantillon (5 folds, ce n'est pas toute la population).
- **`re.fullmatch`** : `fold1_seed0_old` ne compte pas comme `fold1_seed0`.
- **E3 = `resnet18_layer4_rerun`** : la version relancée dans la même session Kaggle que E4 et E5 (section 11).

### Ce qu'on observe (`make aggregate`, relancé : identique à `results/comparison.csv`)
| ID | Macro-F1 | Écart vs réf. | Folds gagnés | Verdict |
|---|---:|---:|---:|---|
| B1 vs B0 | 0,260 | +0,146 | 5 | gagne |
| E1 vs B1 | 0,501 | +0,240 | 5 | **H1 confirmée** |
| E2 vs E1 | 0,507 | +0,007 | 3 | non démontré |
| E3 vs E2 | 0,661 | +0,154 | 5 | **H2 confirmée** |
| E4 vs E3 | 0,654 | −0,007 | 1 | pas de gain de macro-F1 (H3 jugée sur les classes rares, [§10](#s10)) |
| E5 vs E3 | 0,626 | −0,035 | 0 | **H4 confirmée** (les augmentations aident) |

### Ton gain de F1 est-il significatif ? (question d'entretien)
Réponse actuelle (raisonnement, pas un test statistique) : l'écart de H1 (+0,24) et de H2 (+0,15) est bien plus
grand que le bruit mesuré (0,0075 entre graines, 0,024 entre folds) et gagne sur 5 folds sur 5. Celui de H4 (+0,035)
l'est de peu. Les intervalles de confiance par bootstrap sur les lésions arrivent en phase 4 (pas encore codés).

### Pièges
- **`FileNotFoundError: aucune graine complète (5 folds) dans ...`** : un fold manque pour toutes les graines.
- **Comparer des runs de sessions Kaggle différentes** : écart de 0,001 à 0,005 qui ne vient pas du modèle.

---

<a id="s15"></a>
## 15. Voir quelles classes le modèle confond : matrice de confusion et rappel par classe (phase 3 — `scripts/aggregate.py`, `scripts/make_figures.py`)

**Idée :** on compte, pour chaque **vraie** classe, dans quelle classe ses images ont été **prédites**, puis on divise
chaque ligne par son effectif : la diagonale est le **rappel** de la classe, le reste dit où partent les erreurs.

```
E4, 5 folds cumulés, seed 0 (extrait, lignes normalisées)      lecture d'une ligne :
                prédit →  ...    mel     nv    ...             « sur 100 vrais mélanomes,
réel  mel  (n=1001)       ...   0,61    0,19   ...               61 sont reconnus, 19 sont pris pour des nv »
      df   (n=103)        ...           0,07
      nv   (n=6034)       ...           0,85

rappel_c    = cm[c, c] / Σ_j cm[c, j]     (ligne)    ← « je trouve les c »
précision_c = cm[c, c] / Σ_i cm[i, c]     (colonne)  ← « quand je dis c, c'est vrai »
```

### Pourquoi normaliser par ligne ?
Sans normalisation, la matrice est écrasée par `nv` (6 034 images) : les 103 `df` y sont invisibles. Diviser chaque
ligne par son effectif met toutes les classes sur la même échelle [0, 1].

### Pourquoi cumuler les 5 folds ?
Chaque image de `train_val` est dans **exactement un** fold de validation : réunir les 5 `val_logits.csv` donne une
prédiction *out-of-fold* pour chacune des 9 013 images, faite par un modèle qui ne l'a jamais vue. Avec 103 `df`
au lieu de ≈ 20, le rappel est bien moins bruité.

```python
# scripts/make_figures.py
cm = np.zeros((7, 7))
for t, q in zip(y, p):
    cm[t, q] += 1                                    # ligne = vraie classe, colonne = prédite
norm = cm / cm.sum(axis=1, keepdims=True)            # keepdims : (7, 1) pour diviser chaque LIGNE
```
```python
# scripts/aggregate.py — les quatre critères de H3, par fold
def recall(c):
    return ((pred == c) & (y == c)).sum() / (y == c).sum()
n_pred = (pred == 5).sum()
precision_nv = ((pred == 5) & (y == 5)).sum() / n_pred if n_pred else 0.0     # aucun nv prédit : 0
```

### Les astuces du code
- **`keepdims=True`** : `cm.sum(axis=1)` a la forme `(7,)` ; sans `keepdims`, numpy diviserait chaque **colonne**
  (broadcast sur le dernier axe) et la matrice serait fausse sans erreur.
- **`np.add.at(cm, (y, p), 1)`** ferait la même chose que la boucle sans boucle Python ; `cm[y, p] += 1` serait **faux**
  (un seul +1 par case même si plusieurs images y tombent).
- **Les figures se recalculent depuis `results/`** (aucun entraînement) : `python -m scripts.make_figures`.

### Ce qu'on observe (`notes/RESULTS.md`, recalculé)
- Confusions dominantes de E4 : `mel → nv` 0,19, `bkl → mel` 0,14, `bkl → nv` 0,12.
- E3 : `mel → nv` **0,35**, `df → nv` 0,23 ; E4 : 0,19 et 0,07.

---

<a id="s16"></a>
## 16. Savoir combien coûte une époque avant de lancer (phase 0 — `scripts/time_epoch.py`)

**Idée :** on chronomètre quelques lots seulement et on **extrapole** à l'époque entière, pour décider où entraîner
(CPU ou GPU) avant d'y passer des heures.

```
226 lots de 32 (folds 1 à 4)        on en mesure 20  ─▶  s / lot  ─▶  × 226  = durée d'une époque

Machine                              s / lot   époque          20 époques
Kaggle T4                             0,327     1,2 min    ≈   25 min
CPU portable 8 cœurs                  4,68     17,6 min    ≈    6 h        → GPU ≈ 14 × plus rapide
```

```python
class LimitedLoader:
    """Limite un DataLoader à n lots pour mesurer sans parcourir l'époque entière."""
    def __init__(self, loader, max_batches):
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = [None] * min(len(loader.dataset), max_batches * loader.batch_size)   # faux dataset
    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                return
            yield batch
...
if device.type == "cuda":
    torch.cuda.synchronize()                 # attendre la fin réelle des calculs GPU
```

### Les astuces du code
- **`self.dataset = [None] * ...`** : `train_one_epoch` divise par `len(loader.dataset)` ; le faux dataset a la
  bonne longueur pour que la perte affichée reste une moyenne correcte, sans modifier `train_one_epoch`.
- **`torch.cuda.synchronize()`** : les appels CUDA sont **asynchrones** ; sans cette ligne, on mesurerait le temps
  d'envoi des ordres au GPU, pas le temps de calcul.
- **`time.perf_counter()`** : horloge monotone faite pour mesurer des durées.

### Pièges
- Les mesures CPU sur 3 à 8 lots incluent le démarrage des workers : **ordre de grandeur**, pas mesure précise.

---

<a id="s17"></a>
## 17. Chaîne complète : de HAM10000 à `comparison.csv`

```
metadata.csv ─▶ make splits ─▶ make preprocess ─▶ make baselines ─▶ make features ─▶ scripts.train
                 (§2)           (§3)              B0, B1 (§6)       E1 (§8)          E2…E5 : 4 configs × 5 folds
                                                                                     Kaggle T4 (§7, §9–12)
                                                                                            │
          figures/ ◀─ scripts.make_figures (§15) ◀─ results/comparison.csv ◀─ make aggregate (§14)
```

| Étape | Commande | Pourquoi |
|---|---|---|
| Partitions | `make splits` (une seule fois, puis **copier** les CSV) | mêmes folds pour tous ([§2](#s2)) |
| Images 256 px | `make preprocess` (≈ 5 min, relançable) | époque 2 × plus rapide sur GPU ([§3](#s3)) |
| Plancher | `make baselines` | B0 et B1 ([§6](#s6)) |
| Réseau figé | `make features` | E1 sans GPU ([§8](#s8)) |
| Contrôles | `python -m scripts.train --config ... --fold 0 --limit 64` | pipeline qui tourne ([§13](#s13)) |
| Fine-tuning | `python -m scripts.train --config configs/<variante>.yaml --fold k --seed s` | E2 à E5 ([§7](#s7)–[§12](#s12)) |
| Synthèse | `make aggregate` puis `python -m scripts.make_figures` | verdicts H1 à H4 ([§14](#s14)) |
| Tests | `make test` | contrats (pas de fuite, ordre des logits, graines) |

- Les runs écrivent dans `runs/` (avec `best.pt`, non versionné) ; on copie dans `results/` les fichiers légers
  (`metrics.json`, `history.csv`, `config.yaml`, `val_logits.csv`).
- Résultat : macro-F1 de 0,115 (dummy) à **0,661** (E3), H1, H2, H4 confirmées, H3 confirmée sur les classes rares.

---

<a id="s18"></a>
## 18. Rendu : règles du projet et fichiers attendus

### Règles générales (protocole et guide)
- **Le test n'est lu par aucun script avant la phase 6.** Tous les scripts lisent `train_val.csv` et vérifient
  `(train_val["split"] == "train_val").all()`.
- Hyperparamètres fixés **avant** de lancer les 5 folds ; une variante = un seul changement.
- Métrique principale : **macro-F1**, jamais changée. Règle : moyenne meilleure **et** ≥ 4 folds sur 5.
- Partitions **copiées**, pas régénérées. MD5 de référence : `splits.csv` 132c187583dee168d04f11f39d737cb8,
  `train_val.csv` 641df133a9a2e40e9a35b66f224434b4, `test.csv` 35ce9feea522a7784999b36e65427132.
- Projet expérimental : **pas d'usage clinique**.

### Fichiers écrits par chaque lancement (`runs/<config>/fold<k>_seed<s>/`)
| Fichier | Contenu | Sert à |
|---|---|---|
| `config.yaml` | config + fold, graine, commit Git, dossier d'images | traçabilité |
| `history.csv` | époque, pertes, F1, accuracy, balanced accuracy, durée | courbes ([§12](#s12)) |
| `best.pt` | poids de la meilleure époque | test final, démo |
| `val_logits.csv` | `image_id, lesion_id, label, logit_0 … logit_6` | calibration, abstention, bootstrap |
| `metrics.json` | métriques de la meilleure époque | `comparison.csv` |

### Critères de passage par phase (guide)
| Phase | Critère |
|---|---|
| 0 | `make splits`, `make baselines`, `make test` passent |
| 2 | 32 images sur-apprises ; 5 fichiers produits ; relance = même macro-F1 à 1e-4 près |
| 3 | `comparison.csv` contient B0 à E5 ; `RESULTS.md` donne un verdict H1 à H4 |

---

<a id="s19"></a>
## 19. Pièges à retenir (liste de contrôle)

Chaque ligne renvoie vers la solution (clique sur le lien).

- [ ] Labels recalculés avec `sorted(unique())` ou `LabelEncoder` → [solution : liste `CLASSES` figée](#s1)
- [ ] Partition par image : 38 % des images de validation ont une sœur en entraînement → [solution : grouper par `lesion_id`](#s2)
- [ ] Un fold sans `df` → [solution : stratifier en plus de grouper](#s2)
- [ ] Partitions régénérées sur Kaggle, différentes → [solution : copier les CSV, vérifier les MD5](#s18)
- [ ] Un script de dev qui lit `splits.csv` (donc le test) → [solution : `train_val.csv` + `assert`](#s2)
- [ ] `FileNotFoundError: Image introuvable` → [solution : bon `--image-dir`, `make preprocess`](#s3)
- [ ] Index pandas troué après un filtre → [solution : `reset_index(drop=True)`](#s3)
- [ ] `Resize((224, 224))` écrase le 4:3 → [solution : choix assumé, `CenterCrop` en expérience à part](#s4)
- [ ] Augmentations en validation → [solution : `get_val_transforms()`](#s4)
- [ ] Juger à l'accuracy (0,669 pour « toujours nv ») → [solution : macro-F1](#s5)
- [ ] `average="weighted"` au lieu de `"macro"` → [solution : `average="macro"`](#s5)
- [ ] Un modèle « bon » qui ne bat pas la couleur → [solution : baselines B0, B1](#s6)
- [ ] BatchNorm des blocs gelés encore en mode train → [solution : remettre ces BatchNorm en `eval()` + test](#s7)
- [ ] Même learning rate pour `layer4` et la tête → [solution : groupes de paramètres (option)](#s7)
- [ ] Cache de features décalé par rapport à `train_val.csv` → [solution : stocker et comparer les `image_id`](#s8)
- [ ] Augmentations avec un cache de features → [solution : pas d'augmentation pour E1](#s8)
- [ ] Perte d'époque mal moyennée avec un dernier lot incomplet → [solution : `loss.item() * len(labels)`](#s9)
- [ ] Optimiseur et perte inversés par un appel positionnel → [solution : arguments nommés `*`](#s9)
- [ ] Logits non sauvés : calibration impossible sans réentraîner → [solution : `predict` + `val_logits.csv`](#s9)
- [ ] Comparer la perte de validation de E4 aux autres → [solution : comparer la macro-F1](#s9)
- [ ] Poids de classes de taille 6 (classe absente) → [solution : `reindex(range(num_classes))`](#s10)
- [ ] Poids infinis pour une classe vide → [solution : `where(counts > 0)` + `fillna(0)`](#s10)
- [ ] Croire que la pondération améliore la macro-F1 → [solution : lire rappels et précision `nv`](#s10)
- [ ] Augmentations différentes selon le nombre de workers → [solution : `seed_worker` + `Generator`](#s11)
- [ ] Comparer des runs de sessions Kaggle différentes → [solution : même session, ou plusieurs graines](#s11)
- [ ] Sauver les logits de la dernière époque au lieu de la meilleure → [solution : `best` garde ses logits](#s12)
- [ ] S'arrêter sur la perte de validation (minimale trop tôt) → [solution : arrêt sur la macro-F1, patience 4](#s12)
- [ ] Perte initiale mesurée sur un lot trié (1,31 au lieu de 1,95) → [solution : lot mélangé](#s13)
- [ ] Conclure sur un écart de 0,007 avec une seule graine → [solution : appariement, 4/5, plusieurs graines](#s14)
- [ ] Une graine incomplète dans la moyenne → [solution : `seeds_available` (intersection)](#s14)
- [ ] Matrice de confusion normalisée par colonne par erreur → [solution : `keepdims=True`](#s15)
- [ ] `cm[y, p] += 1` qui perd les doublons → [solution : boucle ou `np.add.at`](#s15)
- [ ] Chrono GPU qui s'arrête avant la fin du calcul → [solution : `torch.cuda.synchronize()`](#s16)
