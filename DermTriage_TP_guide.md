# DermTriage — Parcours de travaux pratiques guidés

**Version pédagogique du cahier des charges — 23 septembre 2026**  
**Format :** travail individuel, notebooks Python, étapes courtes, sans corrigé.  
**Point de départ :** tu connais les bases de Python, mais tu débutes dans la construction d'un projet de vision complet. Le TP 0 permet de vérifier ces bases.

> Projet pédagogique sur des données publiques. Les prédictions et les priorités simulées ne constituent ni un diagnostic ni une recommandation de prise en charge.

## 1. Ta mission

Construire progressivement un programme qui reçoit une image dermoscopique et produit des scores pour les catégories du jeu de données. Tu apprendras ensuite à mesurer ses erreurs. Si tu souhaites aller plus loin, tu utiliseras ses scores dans une petite expérience de tri de dossiers fictifs.

Le projet est réussi si tu peux expliquer tes choix, reproduire tes résultats et identifier les limites de ton modèle. **Aucun score minimal n'est imposé.** Un résultat modeste correctement évalué vaut un travail abouti.

Tu disposes des consignes, de quelques rappels et d'indices gradués. Le code, les calculs, les graphiques, les observations et les conclusions sont à produire par toi-même. Les propriétés à vérifier indiquent comment contrôler ton travail ; elles ne donnent pas les résultats des expériences.

### Ce qui change par rapport au cahier des charges initial

| Sujet | Nouvelle place |
|---|---|
| Exploration, séparation des données, modèle simple, évaluation | Socle du parcours |
| Plusieurs architectures et modèles de fondation | Un seul petit réseau préentraîné pour commencer |
| Calibration, masques, simulation | Trois TP d'approfondissement indépendants |
| Prédiction conforme, OOD, Cobham, optimisation d'effectifs | Suites possibles après ce parcours |
| API, Docker, ONNX, MLflow, agent LLM | Retirés des exigences de cette première version |
| Validation smartphone | Retirée : aucun jeu explicitement smartphone dans la liste fournie |
| Planning de deux semaines | Progression par compétences ; tu avances après chaque point de contrôle |

### Organisation du parcours

Les durées sont des estimations de travail personnel, hors installations et longs calculs. Prévois davantage si Python est encore nouveau pour toi.

| Partie | TP | Durée indicative | Résultat attendu |
|---|---|---|---|
| A — Prendre en main les données | 0, 1, 2 | 6–9 h | Données comprises et partitions enregistrées |
| B — Apprendre à construire un modèle | 3, 4, 5, 6 | 12–19 h | Références simples et modèle choisi sur validation |
| C — Approfondir, au choix | A1, A2, A3 | 3–5 h chacun | Masques, calibration ou simulation |
| D — Terminer proprement | 7 | 3–5 h | Évaluation finale et compte rendu |

**Parcours court complet :** TP 0 à 6, puis TP 7, soit environ 21–33 h.  
**Parcours étendu :** insère les approfondissements souhaités avant le TP 7.  
**Rythme possible :** deux séances de deux heures par semaine, avec un petit objectif par séance.

## 2. Les fichiers que tu as à disposition

Seul le cahier des charges a été joint à cette conversation. Les données ci-dessous ont été indiquées par leur nom : leur contenu local, leurs effectifs et leurs colonnes restent à vérifier dans les TP.

| Fichier ou dossier indiqué | Rôle dans ce parcours | Quand l'utiliser |
|---|---|---|
| `HAM10000_images_part_1` | Premier lot d'images HAM10000 | Dès le TP 0 |
| `HAM10000_images_part_2` | Second lot du même jeu | Dès le TP 0 |
| `HAM10000_metadata.csv` | Table reliant notamment images, lésions et catégories ; vérifier le schéma | Dès le TP 1 |
| `HAM10000_segmentations_lesion_tschandl` | Masques de lésions à associer aux images après vérification | TP A1 |
| `ISIC2018_Task3_Test_Images` | Images réservées à l'évaluation finale complémentaire | Audit technique au TP 2 ; prédictions au TP 7 |
| `ISIC2018_Task3_Test_GroundTruth.csv` | Étiquettes de référence du test ; schéma à vérifier | Au TP 7 |
| `ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.csv` | Fichier associé à une étude d'interaction humain–IA ; signification des colonnes à documenter | Bonus documentaire après le parcours |

**Deux précautions de méthode :**

- `part_1` et `part_2` sont des lots de fichiers, pas des partitions d'apprentissage et de test. Tu construiras tes propres partitions à partir des identifiants de lésions.
- Le fichier d'interaction humain–IA ne remplace pas les étiquettes diagnostiques. Il n'entre ni dans les variables d'entrée ni dans l'apprentissage du modèle de ce TP. Ne déduis pas la signification d'une colonne de son nom seul.

HAM10000 et le challenge ISIC 2018 concernent ici des images dermoscopiques [R1, R2]. Le second test n'est donc pas une démonstration de fonctionnement sur des photos de téléphone. Son indépendance exacte doit être documentée.

## 3. Méthode de travail

### À chaque séance

1. Lis l'objectif et reformule-le en une phrase.
2. Écris ce que tu penses observer avant de lancer le calcul.
3. Implémente une petite étape, puis vérifie sa sortie.
4. Consulte l'indice 1 si tu bloques. Essaie avant de lire l'indice 2.
5. Termine par cinq lignes : travail effectué, observation, explication possible, limite, prochaine action.

Si tu demandes de l'aide, joins le numéro de la question, ton essai, la sortie ou l'erreur et ce que tu attendais. Demande d'abord un indice ou une explication du blocage. Une solution complète n'est pas nécessaire pour avancer.

### Outils du socle

Un environnement Python avec Jupyter, pandas, NumPy, Matplotlib, Pillow, scikit-learn, PyTorch et torchvision suffit. Utilise la procédure officielle adaptée à ton système pour installer PyTorch [R4]. Note les versions effectivement installées.

Les premiers TP fonctionnent sur CPU. Pour le réseau, commence par un petit sous-ensemble d'apprentissage. Un GPU accélère les essais, mais tu peux réduire le nombre d'époques ou conserver le réseau gelé si tu n'en as pas. Note toujours la taille du sous-ensemble utilisé ; ne compare pas silencieusement deux budgets de données différents.

| Emplacement à créer | Contenu |
|---|---|
| `data/raw/` | Données originales, conservées intactes |
| `data/processed/` | Index des images, partitions et éventuelles données dérivées |
| `notebooks/` | Un notebook par TP |
| `outputs/figures/` | Figures choisies pour le compte rendu |
| `outputs/models/` | Modèles sauvegardés et correspondance des classes |
| `outputs/metrics/` | Mesures, prédictions et journal d'expériences |
| `README.md` | Instructions et présentation du projet |

Ne publie pas les images dans ton dépôt. Relève les sources et les conditions d'utilisation des fichiers que tu as téléchargés, sans supposer que tous ont la même licence.

---

## Partie A — Prendre en main les données

## TP 0 — Installer et manipuler une image

**Durée : 1–2 h. Objectif :** pouvoir ouvrir une table et une image, puis relancer le notebook sans dépendre d'une cellule exécutée auparavant.

### Travail demandé

1. Crée les dossiers de travail. Si tes données sont encore compressées, extrais-les en conservant les archives originales.
2. Définis un emplacement racine unique pour les données, afin d'éviter de répéter les chemins dans chaque cellule.
3. Affiche les versions de Python et des bibliothèques utilisées. Vérifie si PyTorch détecte un GPU.
4. Charge les métadonnées et affiche quelques lignes, les noms de colonnes et leurs types.
5. Ouvre une image depuis chaque lot HAM10000. Affiche leurs dimensions, leur mode de couleur et leur représentation sous forme de tableau.
6. Affiche une image avec son identifiant, sans essayer de poser un diagnostic.
7. Redémarre le noyau et exécute toutes les cellules dans l'ordre.

### Questions

- Quelle différence fais-tu entre un chemin de fichier, une image et un tableau de pixels ?
- Que représentent les dimensions du tableau que tu observes ?
- Quelle information manque-t-il pour relier une image à sa ligne de métadonnées ?

**Indice 1 :** sépare la recherche d'un fichier de son ouverture.  
**Indice 2 :** consulte `pathlib.Path`, `pandas.read_csv` et `PIL.Image.open` ; explore les attributs avant d'écrire une fonction complexe.

**À rendre :** `00_demarrage.ipynb`, deux images affichées et les réponses aux questions.  
**Point de contrôle :** le notebook fonctionne depuis un noyau vide ; les chemins sont centralisés. Si les boucles, fonctions ou indexations te bloquent, fais un exercice court sur ces notions avant le TP 1.

## TP 1 — Comprendre et relier les données

**Durée : 2–3 h. Objectif :** construire une table fiable, avec une ligne par image utilisable.

### Travail demandé

1. Établis un dictionnaire des colonnes du CSV : nom, type, signification documentée, valeurs manquantes.
2. Compte les fichiers images dans chacun des deux lots. Recherche les identifiants répétés entre lots avant de construire ton index.
3. Construis une correspondance entre identifiant d'image et chemin réel. Une extension ou une arborescence ne doit pas être supposée sans inspection.
4. Relie cet index aux métadonnées. Liste les images sans ligne de métadonnées et les lignes sans image.
5. Compare le nombre d'images au nombre de lésions distinctes. Étudie combien d'images appartiennent à chaque lésion.
6. Recherche les lésions associées à plusieurs étiquettes. Si tu en trouves, consigne le problème et documente son traitement avant le découpage.
7. Produis un tableau des effectifs par catégorie, puis un diagramme en barres. Affiche une petite grille d'exemples reproductible.
8. Enregistre l'index dans `data/processed/image_index.csv`, avec au minimum identifiant d'image, identifiant de lésion, catégorie et chemin.

### Questions

- Le nombre d'images et le nombre de lésions mesurent-ils la même chose ?
- Toutes les catégories ont-elles autant d'exemples ? Quelle conséquence anticipes-tu ?
- Comment distingues-tu une image absente d'une image impossible à décoder ?

**Indice 1 :** pose séparément les questions d'unicité, de correspondance et de valeurs manquantes.  
**Indice 2 :** explore `value_counts`, `nunique`, `groupby`, `merge` et les contrôles de cardinalité d'une jointure. N'écrase pas silencieusement un identifiant dupliqué dans un dictionnaire.

**À rendre :** `01_exploration.ipynb`, l'index, un graphique et une liste des anomalies.  
**Point de contrôle :** chaque ligne retenue mène à une image lisible et à une étiquette ; les exclusions sont comptées et justifiées.

## TP 2 — Séparer les données sans fuite

**Durée : 3–4 h. Objectif :** préparer une évaluation sur des lésions non utilisées pour apprendre.

### Repère de cours

L'entraînement ajuste les poids. La validation sert à choisir les réglages. Le test mesure le résultat après ces choix. Plusieurs images d'une même lésion sont liées [R1] : la séparation doit respecter ce groupe.

### Travail demandé

1. Rédige, avant de coder, la règle qui décide quelles images doivent rester ensemble.
2. Construis des partitions HAM10000 visant environ 70 % / 15 % / 15 % pour entraînement, validation et test interne. Ces proportions sont approximatives ; la séparation des lésions est prioritaire.
3. Utilise une méthode tenant compte des groupes. Compare les répartitions des classes, sans chercher une graine donnant de meilleurs scores.
4. Calcule les intersections d'identifiants de lésions entre les trois partitions. Transforme ce contrôle en assertion.
5. Vérifie qu'aucune image n'est perdue ni ajoutée lors du découpage. Compte images et lésions par classe et par partition.
6. Fixe une graine et enregistre `splits.csv`. Réutilise ensuite ce fichier dans tous les TP.
7. Réalise un audit technique entre HAM10000 et les images ISIC : identifiants communs, fichiers identiques par empreinte, anomalies. Un contrôle de similarité visuelle est un bonus. N'utilise pas les étiquettes ISIC à ce stade.
8. Définis à l'avance le traitement des éventuels chevauchements pour l'évaluation finale. Conserve le test officiel intact et consigne séparément un sous-ensemble sans chevauchement détecté si nécessaire.

**Si tu choisis le TP A2 :** réserve dès maintenant une quatrième partition de calibration, par exemple en visant 60 % / 15 % / 10 % / 15 % pour entraînement, validation, calibration et test interne. Elle ne participe ni à l'entraînement ni au choix du réseau. Les proportions sont des consignes de travail, pas des résultats attendus.

### Questions

- Quel risque introduit un découpage aléatoire image par image ?
- Une séparation par lésion prouve-t-elle une séparation par patient si l'identifiant patient n'est pas disponible ?
- L'absence d'identifiants ou de fichiers identiques prouve-t-elle l'absence de toutes les images apparentées ?

**Indice 1 :** raisonne d'abord sur la liste des groupes, puis sur les images qui leur appartiennent.  
**Indice 2 :** compare les possibilités de `GroupShuffleSplit` et `StratifiedGroupKFold` [R3]. Une stratification avec groupes peut rester approximative ; documente les effectifs obtenus, notamment pour les catégories rares.

**À rendre :** `02_partitions.ipynb`, `splits.csv` et un tableau des contrôles.  
**Point de contrôle :** intersections de lésions nulles, affectations persistantes et répartition documentée. Le test interne et le test ISIC sont désormais réservés au TP 7. Les transformations apprises et les poids de classes seront calculés sur l'entraînement uniquement.

---

## Partie B — Construire et améliorer un modèle

## TP 3 — Établir une référence simple

**Durée : 2–3 h. Objectif :** savoir à quoi comparer un réseau avant d'en entraîner un.

### Travail demandé

1. Choisis un ordre explicite pour les sept catégories rencontrées et enregistre le dictionnaire catégorie–indice. Vérifie qu'il est réutilisé partout.
2. Construis un classifieur qui prédit toujours la catégorie majoritaire de l'entraînement.
3. Évalue cette référence sur la validation : accuracy, balanced accuracy, macro-F1 et matrice de confusion.
4. Ajoute un modèle simple à partir de quelques caractéristiques d'image que tu définis : par exemple des statistiques de couleur. Décris les informations qu'elles conservent et celles qu'elles perdent.
5. Entraîne une régression logistique sur ces caractéristiques. Si tu standardises les variables, apprends les paramètres de standardisation sur l'entraînement.
6. Compare les deux modèles avec les mêmes métriques et les mêmes images de validation.

### Questions

- Quelle métrique paraît la plus flatteuse pour le modèle majoritaire ? Pourquoi ?
- Que se passe-t-il pour une catégorie jamais prédite ?
- Une différence entre les deux modèles suffit-elle à affirmer que l'un comprend les lésions ?

**Indice 1 :** une référence utile peut être volontairement très simple.  
**Indice 2 :** consulte `DummyClassifier`, `Pipeline`, `StandardScaler` et `LogisticRegression`. Prévois explicitement le cas des métriques indéfinies et indique comment tu les rapportes.

**À rendre :** `03_references.ipynb`, deux matrices de confusion et un tableau de comparaison.  
**Point de contrôle :** aucun calcul d'apprentissage n'utilise la validation ; les sept catégories apparaissent dans les sorties même si certaines ne sont jamais prédites.

## TP 4 — Préparer un chargement d'images pour PyTorch

**Durée : 3–4 h. Objectif :** produire des lots d'images et d'étiquettes corrects.

### Repères de cours

Un `Dataset` décrit comment récupérer un exemple. Un `DataLoader` rassemble des exemples en lots. Une transformation modifie leur représentation. Tu n'as pas encore besoin d'un entraînement long.

### Travail demandé

1. Définis le contrat de ton jeu de données : informations reçues à la création et informations renvoyées pour un exemple.
2. Implémente un `Dataset` fondé sur l'index et les partitions sauvegardées.
3. Prépare une transformation déterministe compatible avec le réseau préentraîné que tu utiliseras au TP 5, en consultant ses poids et son prétraitement officiel.
4. Ajoute au plus deux augmentations simples à l'entraînement. La validation et les tests restent déterministes.
5. Affiche une image avant et après transformation ; tiens compte de la normalisation pour la visualiser.
6. Charge un petit lot. Inspecte dimensions, types, étiquettes et identifiants d'images.
7. Vérifie que les transformations n'altèrent pas les données originales et que les étiquettes restent associées aux bonnes images.

### Questions

- Pourquoi ne pas appliquer des transformations aléatoires à la validation de ce TP ?
- Que représente chaque dimension d'un lot ?
- Quel effet une modification excessive des couleurs pourrait-elle avoir ?

**Indice 1 :** fais fonctionner un exemple, puis quatre, avant le jeu entier.  
**Indice 2 :** explore `__len__`, `__getitem__` et `DataLoader` [R4]. Vérifie séparément l'ordre des axes, le type des pixels et le codage des classes.

**À rendre :** `04_chargement.ipynb` et une grille d'images transformées.  
**Point de contrôle :** pour des images RGB, la forme du lot est `(B, 3, H, W)` ; les cibles sont des indices valides ; deux passages en validation retrouvent le même prétraitement pour chaque image.

## TP 5 — Entraîner un premier réseau par transfert

**Durée : 4–6 h. Objectif :** adapter un modèle existant sans construire une architecture complète.

### Travail demandé

1. Charge un petit réseau préentraîné, par exemple ResNet-18 dans torchvision. Identifie l'extracteur de caractéristiques et la couche de classification [R5].
2. Détermine la dimension de sortie nécessaire pour cette tâche. Remplace la couche finale.
3. Gèle d'abord l'extracteur et n'entraîne que la tête. Vérifie quels paramètres restent entraînables et maîtrise le mode des couches de normalisation du corps gelé.
4. Définis la perte adaptée à une classification exclusive en sept catégories. Vérifie ce qu'elle attend comme entrées et cibles.
5. Écris une boucle pour une époque d'entraînement, puis une fonction d'évaluation distincte.
6. Effectue un essai sur un très petit lot d'entraînement. Vérifie que le modèle peut apprendre ces exemples avant de lancer l'expérience complète.
7. Lance un budget court annoncé à l'avance, par exemple cinq époques. Trace les pertes et la macro-F1 de validation.
8. Sauvegarde le modèle choisi selon la macro-F1 de validation, sa configuration, le dictionnaire des classes et le prétraitement.
9. Recharge-le dans une session neuve et vérifie ses prédictions sur les mêmes exemples de validation.

### Questions

- Que signifie « préentraîné » ? Quelles parties apprennent réellement ici ?
- Quelle différence y a-t-il entre un score brut, une probabilité et une classe prédite ?
- Pourquoi sélectionner une époque sur la validation et non sur le test ?

**Indice 1 :** vérifie d'abord formes, perte, gradients et paramètres mis à jour ; augmenter le nombre d'époques ne résout pas un problème de câblage.  
**Indice 2 :** consulte les contrats de `CrossEntropyLoss`, `train()`, `eval()` et du contexte sans gradients. Évite de transformer les sorties avant la perte sans vérifier ce qu'elle attend.

**À rendre :** `05_transfert.ipynb`, courbes, modèle et configuration.  
**Point de contrôle :** perte finie, paramètres attendus effectivement mis à jour, validation sans apprentissage et rechargement fonctionnel. Il n'est pas obligatoire de dépasser la référence pour valider l'implémentation ; une contre-performance doit être examinée.

## TP 6 — Mener une expérience contrôlée

**Durée : 3–6 h. Objectif :** améliorer ou comprendre le modèle en ne changeant qu'une chose à la fois.

### Travail demandé

1. Écris une hypothèse testable à partir du TP 5.
2. Choisis **une seule** modification : pondération de la perte, augmentation différente ou dégel du dernier bloc du réseau. N'ajoute pas trois techniques simultanément.
3. Fixe les autres paramètres : partitions, budget, critère de sélection et prétraitement d'évaluation.
4. Si tu utilises des poids de classes, calcule-les uniquement sur l'entraînement. Si tu dégèles un bloc, justifie ton taux d'apprentissage.
5. Compare expérience de référence et variante sur la validation. Rapporte les métriques globales et le rappel de chaque catégorie.
6. Examine au moins huit erreurs de validation selon une règle d'échantillonnage annoncée. Montre aussi quelques réussites.
7. Si le temps le permet, répète les deux expériences avec une autre graine. Sinon, mentionne que la variabilité n'est pas estimée.
8. Rédige une décision : conserver la référence ou la variante, avec une justification fondée sur la validation.

### Questions

- Le gain est-il partagé par toutes les catégories ?
- Une baisse de la perte d'entraînement implique-t-elle un meilleur résultat sur la validation ?
- Quelles observations feraient abandonner ton hypothèse ?

**Indice 1 :** une expérience qui n'améliore rien peut répondre correctement à une question.  
**Indice 2 :** crée un journal avec identifiant d'expérience, modification, graine, données utilisées, budget, métriques et décision. Reste à deux ou trois variantes au total pour ce parcours.

**À rendre :** `06_experiences.ipynb`, journal d'expériences et conclusion d'une demi-page.  
**Point de contrôle :** chaque comparaison est traçable et le choix du modèle n'utilise aucun test. Choisis maintenant tes approfondissements, ou passe directement au TP 7.

---

## Partie C — Approfondissements facultatifs

Ces TP se réalisent avant l'évaluation finale. Ils ne sont pas nécessaires pour terminer le parcours court. Tu peux en choisir un seul.

## TP A1 — Utiliser les masques pour explorer le rôle du fond

**Durée : 3–5 h. Prérequis :** TP 4 à 6. **Objectif :** réaliser une expérience d'occultation avec les masques déjà fournis, sans entraîner un réseau de segmentation.

### Travail demandé

1. Inspecte les noms, dimensions et valeurs des masques. Construis la correspondance image–masque à partir des identifiants.
2. Vérifie visuellement la superposition sur plusieurs exemples et calcule le taux de correspondance.
3. Choisis un sous-ensemble de validation avec masques disponibles. Annonce sa composition par catégorie.
4. Prépare trois versions de chaque image : originale, région de lésion conservée seule, fond conservé seul. Décris la valeur de remplacement choisie.
5. Applique le même modèle gelé aux trois versions, avec les mêmes règles de prétraitement.
6. Compare les scores et les métriques sur exactement les mêmes exemples.
7. Formule une conclusion prudente et propose une expérience complémentaire.

### Questions

- Un masque binaire et une catégorie diagnostique sont-ils le même type d'annotation ?
- Si les scores changent beaucoup, quelles explications sont possibles ?
- Cette expérience prouve-t-elle que le modèle utilisait un raccourci lors de l'apprentissage ?

**Indice 1 :** commence par prouver l'alignement des pixels avant d'interpréter les prédictions.  
**Indice 2 :** le redimensionnement d'un masque catégoriel doit préserver ses catégories ; étudie l'interpolation au plus proche voisin. Synchronise toutes les transformations géométriques entre image et masque.

**À rendre :** `A1_masques.ipynb`, exemples superposés, tableau apparié et limites.  
**Point de contrôle :** aucun masque n'est décalé ; les trois conditions utilisent les mêmes images. Le masquage crée aussi un changement de distribution : une baisse de performance n'est pas une preuve causale d'un raccourci. Les masques manuels restent un outil d'analyse, pas une entrée implicitement disponible pour toute future image [R6].

## TP A2 — Explorer la confiance et la calibration

**Durée : 3–5 h. Prérequis :** modèle gelé et partition de calibration réservée au TP 2. **Objectif :** comparer confiance annoncée et fréquence de réussite.

### Repère de cours

La calibration concerne l'accord entre probabilités et fréquences observées. Elle se distingue de la capacité à classer les exemples. Une température scalaire ajuste l'échelle des scores bruts avant conversion en probabilités ; elle n'entraîne pas à nouveau le réseau.

### Travail demandé

1. Extrais les scores bruts sur la calibration et conserve les étiquettes correspondantes.
2. Définis la confiance comme la probabilité de la catégorie prédite. Choisis à l'avance des intervalles de confiance, par exemple cinq intervalles fixes.
3. Pour chaque intervalle, calcule effectif, confiance moyenne et fréquence de prédictions correctes. Traite explicitement les intervalles vides.
4. Trace un diagramme de fiabilité descriptif sur la calibration.
5. Recherche le principe du temperature scaling [R7]. Définis une température strictement positive et ajuste-la sur cette partition seulement, selon la log-vraisemblance négative.
6. Sauvegarde cette température avec le modèle. Prépare la comparaison avant/après qui sera appliquée au test au TP 7.

### Questions

- Pourquoi les valeurs mesurées sur la partition ayant servi à ajuster la température ne sont-elles pas l'évaluation finale ?
- Un intervalle contenant peu d'images permet-il une conclusion solide ?
- La calibration donne-t-elle une garantie sur chaque prédiction individuelle ?

**Indice 1 :** sépare clairement « ajuster la température » et « mesurer son effet ».  
**Indice 2 :** travaille à partir des scores bruts, vérifie la positivité du paramètre et contrôle la somme des probabilités.

**À rendre :** `A2_calibration.ipynb`, diagramme descriptif et température sauvegardée.  
**Point de contrôle :** modèle et température appris sur des partitions différentes ; aucun réglage à partir des résultats de test. Si tu n'avais pas réservé de calibration, reconstruis le protocole et réentraîne avant de poursuivre ; ne prélève pas dans un test déjà consulté.

## TP A3 — Simuler un tri simple de dossiers fictifs

**Durée : 3–5 h. Prérequis :** modèle choisi et probabilités de validation disponibles. **Objectif :** étudier l'effet d'un ordre de lecture dans un scénario volontairement simple.

### Convention pédagogique fournie

Pour conserver le lien avec le projet initial, appelle « groupe prioritaire du TP » les catégories `mel`, `bcc`, `akiec`, et « autres catégories » les catégories restantes. Il s'agit d'une convention expérimentale, pas d'une règle médicale. Construis le score de priorité à partir des probabilités de ces catégories, en utilisant ton dictionnaire de classes.

### Scénario imposé pour commencer

Un lot de dossiers est disponible à l'instant zéro. Un seul lecteur fictif les traite successivement ; chaque lecture dure cinq minutes. Compare un ordre d'arrivée tiré au hasard, puis fixé, à un ordre décroissant de score. La durée et le scénario sont des hypothèses d'exercice.

### Travail demandé

1. Construis les dossiers à partir de prédictions de validation, en conservant au plus une image par lésion selon une règle fixe.
2. Écris les entrées et sorties attendues du simulateur : ordre, début de lecture, fin de lecture et attente pour chaque dossier.
3. Vérifie les calculs à la main sur quatre dossiers inventés avant d'utiliser les prédictions.
4. Compare les deux politiques sur exactement les mêmes dossiers. Fixe une règle de départage des scores égaux.
5. Calcule l'attente moyenne et le P90 pour le groupe prioritaire et les autres catégories. Distingue attente avant lecture et délai jusqu'à la fin de lecture.
6. Répète l'expérience sur plusieurs tirages reproductibles de dossiers de validation, sans modifier la politique après inspection du test.
7. Prépare une politique gelée à évaluer au TP 7. Si tu ajoutes des arrivées successives en bonus, un dossier ne peut être sélectionné avant son arrivée.

### Questions

- Quel groupe gagne ou perd du temps ?
- Le temps nécessaire pour terminer tout le lot change-t-il ?
- Pourquoi cette simulation ne suffit-elle pas à conclure à un bénéfice pour des patients ?
- La proportion de catégories dans HAM10000 décrit-elle nécessairement celle d'un service réel ?

**Indice 1 :** commence par une liste ordonnée et une horloge ; SimPy n'est pas nécessaire pour ce scénario.  
**Indice 2 :** vérifie d'abord l'absence de dossier perdu, dupliqué ou traité simultanément. Si tous les scores sont égaux, la règle de départage doit être explicite.

**À rendre :** `A3_tri.ipynb`, cas manuel, tableau des délais et hypothèses.  
**Point de contrôle :** les vraies catégories servent à mesurer les résultats, jamais à choisir l'ordre par score. Les politiques traitent les mêmes dossiers et tu rapportes aussi l'effet sur les autres catégories. Les conclusions concernent uniquement ce lot synthétique.

---

## Partie D — Évaluer et présenter le travail

## TP 7 — Ouvrir les tests et rédiger le bilan

**Durée : 3–5 h. Prérequis :** toutes les décisions expérimentales sont prises. **Objectif :** mesurer le résultat final sans poursuivre le réglage du modèle.

### Avant d'ouvrir les résultats

Consigne la version du modèle, le prétraitement, l'ordre des classes, les métriques et, le cas échéant, la température et la politique de tri. Prépare le script d'évaluation. Les exemples de test ne doivent pas servir à choisir ces paramètres.

### Travail demandé

1. Évalue le modèle retenu sur le test interne HAM10000 : accuracy, balanced accuracy, macro-F1, rappel par catégorie et matrice de confusion. Tu peux évaluer les références déjà figées sans choisir un nouveau modèle à partir du test.
2. Charge le CSV de vérité terrain ISIC. Inspecte son schéma, ses valeurs et son encodage des étiquettes ; ne présume ni d'un format one-hot valide ni de l'absence de lignes inconnues ou incomplètes.
3. Établis et vérifie une correspondance explicite entre ses catégories et celles de HAM10000. Ne te contente pas de convertir la casse des noms.
4. Relie les images et les étiquettes par identifiant, jamais par ordre des fichiers ou numéro de ligne. Vérifie la cardinalité de la jointure et compte chaque exclusion.
5. Applique les règles de chevauchement définies au TP 2 et évalue le même modèle gelé sur le test complémentaire ISIC. Rapporte les effectifs réellement évalués et les limites de l'audit.
6. Présente les résultats internes et ISIC dans des tableaux séparés. Explique ce qui peut être comparé et ce qui reste incertain.
7. Si tu as réalisé A2, mesure avant/après calibration sur le test interne, avec les intervalles fixés et la log-vraisemblance négative. Une amélioration n'est pas imposée.
8. Si tu as réalisé A3, applique la politique gelée à des prédictions de test interne et rapporte les délais des deux groupes sous les hypothèses annoncées.
9. Examine quelques erreurs finales pour les décrire, sans retourner optimiser le modèle. Un nouveau cycle après lecture des tests doit être annoncé comme exploratoire ; ces tests ne sont plus vierges.
10. Rédige un compte rendu de trois à cinq pages et vérifie qu'un lecteur peut relancer ton parcours depuis le README.

### Questions

- Une différence entre les tests peut-elle venir de la composition des catégories ?
- Que peux-tu affirmer sur de nouvelles lésions ? Sur des photos smartphone ?
- Quelles limites viennent du modèle, des données ou du protocole ?

**Indice 1 :** vérifie les correspondances de classes avant d'interpréter un score étonnant.  
**Indice 2 :** produis un contrôle de jointure avec les identifiants non associés de chaque côté. Si une métrique ne peut pas être calculée, indique la raison au lieu d'inventer une valeur.

**À rendre :** `07_evaluation_finale.ipynb`, tableaux finaux, README et rapport.  
**Point de contrôle :** chaque résultat est relié à un modèle, une partition et un effectif. ISIC est présenté comme test complémentaire dermoscopique, pas comme preuve de généralisation clinique ou smartphone.

### Trame du rapport à compléter

1. **Question étudiée et périmètre :** ce que ton programme fait et les approfondissements choisis.
2. **Données et protocole :** sources, unités image/lésion, exclusions, partitions et contrôle des chevauchements.
3. **Méthodes :** références, réseau, budget et critère de sélection.
4. **Résultats :** tableaux, figures utiles, erreurs et expériences non concluantes.
5. **Limites et suite :** incertitude, biais possibles, absence de validation clinique et prochaine expérience raisonnable.

## 4. Grille d'autoévaluation du socle

| Critère | Points indicatifs | Preuve à fournir |
|---|---:|---|
| Compréhension et intégrité des données | 4 | Index, effectifs, anomalies documentées |
| Séparation et absence de fuite détectée | 5 | Partitions sauvegardées, assertions, audit |
| Construction et comparaison des modèles | 4 | Références, modèle rechargé, expérience contrôlée |
| Évaluation et interprétation | 4 | Métriques adaptées, tests réservés, analyse des erreurs |
| Reproductibilité et clarté | 3 | Notebooks relançables, versions, README et rapport |
| **Total** | **20** | La note ne dépend pas d'un seuil de performance |

Les approfondissements enrichissent le projet sans compenser un problème de fuite. Si un contrôle échoue, corrige le protocole et refais les expériences concernées avant d'interpréter leurs scores.

### Checklist de fin

- [ ] Je peux expliquer la différence entre image, lésion et patient.
- [ ] Mes images sont reliées aux étiquettes par identifiant.
- [ ] Les lésions de mes partitions HAM10000 ne se chevauchent pas.
- [ ] Mes tests n'ont pas servi à choisir les réglages.
- [ ] J'ai comparé mon réseau à des références simples.
- [ ] J'ai présenté les catégories difficiles, pas seulement une moyenne.
- [ ] Je peux recharger le modèle avec le bon prétraitement et les bonnes classes.
- [ ] Je distingue clairement résultat mesuré, hypothèse et limite.
- [ ] J'ai cité les données et vérifié leurs conditions d'utilisation.
- [ ] Mon README indique le caractère pédagogique et l'absence d'usage médical.

## 5. Suites possibles après ce parcours

Choisis une seule suite selon ce que tu veux apprendre : une démo locale pour travailler l'interface ; une segmentation apprise pour approfondir la vision ; une simulation avec arrivées pour la recherche opérationnelle ; ou une étude de l'incertitude pour la statistique.

Le CSV `ISIC2018_Task3_Test_NatureMedicine_AI_Interaction_Benefit.csv` peut servir à un **bonus de lecture critique** : retrouver la documentation associée [R6], identifier l'unité d'observation et les conditions d'assistance, puis proposer une question descriptive. Avant tout calcul, vérifier si les mêmes images ou lecteurs apparaissent plusieurs fois. Sans schéma documenté, ne pas inventer de comparaison « humain contre IA » ni réutiliser les colonnes comme entrées du classifieur.

La prédiction conforme, l'API, Docker et l'agent LLM deviennent des projets suivants. Tu disposeras alors d'un modèle et d'une évaluation que tu comprends.

## 6. Ressources à consulter au moment utile

Ces références servent à comprendre un concept ou une interface. Elles ne constituent pas un corrigé du TP. Le parcours ci-dessus est une adaptation pédagogique du cahier des charges fourni, pas la reproduction d'un tutoriel existant.

- **[R1] Données HAM10000 et images multiples par lésion :** Tschandl et al., [publication du jeu de données](https://www.nature.com/articles/sdata2018161). Utiliser au TP 1 pour documenter les colonnes et catégories.
- **[R2] Cadre du challenge :** [ISIC 2018](https://challenge.isic-archive.com/landing/2018/). Utiliser pour situer la tâche de classification.
- **[R3] Découpage par groupes :** [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) et [guide de validation croisée](https://scikit-learn.org/stable/modules/cross_validation.html). Lire au TP 2.
- **[R4] Bases PyTorch :** [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro) et [chargement de données](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial). Lire les sections correspondant à ton blocage aux TP 4 et 5.
- **[R5] Principe du transfert :** [tutoriel officiel PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial). Comparer les notions de réseau gelé et de fine-tuning ; rédiger ensuite ta propre implémentation.
- **[R6] Masques et collaboration humain–IA :** [dépôt des auteurs](https://github.com/ptschandl/HAM10000_dataset) et [Tschandl et al., Nature Medicine, 2020](https://www.nature.com/articles/s41591-020-0942-0). À utiliser pour documenter les fichiers complémentaires, sans supposer que leur structure locale a été vérifiée.
- **[R7] Calibration :** Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). Lecture ciblée sur la température scalaire au TP A2.

**Première séance :** fais uniquement le TP 0. Ton objectif est d'ouvrir les métadonnées, d'afficher deux images et de relancer le notebook. Tu n'as pas besoin de comprendre tout le projet pour commencer.
