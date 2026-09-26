# DermTriage — Suivi des packages `src/`

**Date :** 26 septembre 2026  
**Auteur :** Lamine SOW  

---

## 1. Structure

src/
├── init.py
├── dataset.py
├── transforms.py
├── train.py
└── utils.py


---

## 2. Description des modules

### `dataset.py` — `DermDataset`

Classe PyTorch `Dataset` qui charge les images à la volée depuis le disque.

**Paramètres :**
- `df` : DataFrame avec les colonnes `image_id` et `label`
- `data_dir` : chemin vers le dossier `data/raw/`
- `transform` : transformation torchvision à appliquer (optionnel)

**Comportement :**
- Reconstruit les paths depuis les deux lots (`part_1`, `part_2`) au moment de l'init
- Lève une `FileNotFoundError` explicite si une image est introuvable
- Applique `reset_index` pour éviter les problèmes d'index après un subset

---

### `transforms.py` — Transformations

Deux fonctions qui retournent un `transforms.Compose` prêt à l'emploi.

| Fonction | Usage | Augmentations |
|---|---|---|
| `get_train_transforms()` | Entraînement | Flip H/V, ColorJitter, Normalize ImageNet |
| `get_val_transforms()` | Validation / Test | Normalize ImageNet uniquement |

**Normalisation ImageNet :**
- mean : `[0.485, 0.456, 0.406]`
- std  : `[0.229, 0.224, 0.225]`

---

### `utils.py` — Fonctions utilitaires

| Fonction | Description |
|---|---|
| `get_class_weights(df, num_classes=7)` | Poids inversement proportionnels à la fréquence des classes. Accepte un DataFrame ou un tensor de labels. |
| `save_model(model, path, config)` | Sauvegarde `state_dict` + dict de config dans un `.pt` |
| `load_model(model, path, device)` | Recharge les poids dans un modèle existant, retourne le modèle |
| `compute_metrics(y_true, y_pred)` | Retourne `accuracy`, `balanced_accuracy`, `f1` (macro) |

**Pourquoi `balanced_accuracy` et `f1` macro ?**  
Le dataset est très déséquilibré (`nv` = 67%). L'accuracy seule est trompeuse —
un modèle qui prédit toujours `nv` obtiendrait 67% sans rien apprendre.

---

### `train.py` — Boucle d'entraînement

| Fonction | Description |
|---|---|
| `train_one_epoch(model, loader, criterion, optimizer, device)` | Une epoch complète avec backprop. Retourne la loss moyenne. |
| `evaluate(model, loader, criterion, device)` | Évaluation sans gradient. Retourne `loss`, `accuracy`, `balanced_accuracy`, `f1`. |

---

## 3. Résultats des tests

```bash
(.venv) zeyrox@zeyrox-Latitude-7420:~/Bureau/project/DermTriage$ python -m pytest tests/ -v
=============================================== test session starts ================================================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /home/zeyrox/Bureau/project/DermTriage/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/zeyrox/Bureau/project/DermTriage
plugins: anyio-4.15.1
collected 31 items                                                                                                 

tests/test_dataset.py::test_len PASSED                                                                       [  3%]
tests/test_dataset.py::test_getitem_retourne_tensor_et_label_correct[0] PASSED                               [  6%]
tests/test_dataset.py::test_getitem_retourne_tensor_et_label_correct[1] PASSED                               [  9%]
tests/test_dataset.py::test_getitem_retourne_tensor_et_label_correct[2] PASSED                               [ 12%]
tests/test_dataset.py::test_getitem_retourne_tensor_et_label_correct[3] PASSED                               [ 16%]
tests/test_dataset.py::test_getitem_retourne_tensor_et_label_correct[4] PASSED                               [ 19%]
tests/test_dataset.py::test_getitem_retourne_tensor_et_label_correct[5] PASSED                               [ 22%]
tests/test_dataset.py::test_getitem_retourne_tensor_et_label_correct[6] PASSED                               [ 25%]
tests/test_dataset.py::test_image_chargee_correspond_a_la_bonne_ligne PASSED                                 [ 29%]
tests/test_dataset.py::test_image_absente_leve_une_erreur_explicite PASSED                                   [ 32%]
tests/test_dataset.py::test_repertoire_absent_leve_une_erreur_explicite PASSED                               [ 35%]
tests/test_train.py::test_train_retourne_loss_float_finie PASSED                                             [ 38%]
tests/test_train.py::test_train_met_a_jour_les_parametres PASSED                                             [ 41%]
tests/test_train.py::test_evaluate_retourne_les_metriques_sans_modifier_les_poids PASSED                     [ 45%]
tests/test_train.py::test_loss_moyenne_par_exemple[train] PASSED                                             [ 48%]
tests/test_train.py::test_loss_moyenne_par_exemple[eval] PASSED                                              [ 51%]
tests/test_transforms.py::test_transformation_callable[get_train_transforms] PASSED                          [ 54%]
tests/test_transforms.py::test_transformation_callable[get_val_transforms] PASSED                            [ 58%]
tests/test_transforms.py::test_forme_type_et_intervalle[get_train_transforms] PASSED                         [ 61%]
tests/test_transforms.py::test_forme_type_et_intervalle[get_val_transforms] PASSED                           [ 64%]
tests/test_transforms.py::test_validation_utilise_la_normalisation_imagenet[0] PASSED                        [ 67%]
tests/test_transforms.py::test_validation_utilise_la_normalisation_imagenet[255] PASSED                      [ 70%]
tests/test_transforms.py::test_validation_deterministe PASSED                                                [ 74%]
tests/test_transforms.py::test_image_source_inchangee[get_train_transforms] PASSED                           [ 77%]
tests/test_transforms.py::test_image_source_inchangee[get_val_transforms] PASSED                             [ 80%]
tests/test_utils.py::test_class_weights_forme_et_valeurs PASSED                                              [ 83%]
tests/test_utils.py::test_class_weights_inversement_proportionnels PASSED                                    [ 87%]
tests/test_utils.py::test_save_model_cree_un_fichier PASSED                                                  [ 90%]
tests/test_utils.py::test_load_model_restaure_poids_et_predictions PASSED                                    [ 93%]
tests/test_utils.py::test_compute_metrics_predictions_parfaites PASSED                                       [ 96%]
tests/test_utils.py::test_compute_metrics_predictions_imparfaites PASSED                                     [100%]

================================================ 31 passed in 9.50s ================================================
(.venv) zeyrox@zeyrox-Latitude-7420:~/Bureau/project/DermTriage$ 
```