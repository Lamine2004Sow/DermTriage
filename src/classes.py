"""Liste figée des classes : l'indice dans CLASSES est le label entier.

Ne jamais recalculer ce mapping à partir des données (LabelEncoder,
sorted(unique())) : une classe absente d'un sous-ensemble décalerait les labels.
"""

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}
