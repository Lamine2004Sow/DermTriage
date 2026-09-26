import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class DermDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Répertoire de données introuvable : {self.data_dir}"
            )
        # Reconstruction des paths à partir des image_id
        self.chemins = {
            f.stem: f
            for dossier in [
                self.data_dir / "HAM10000_images_part_1",
                self.data_dir / "HAM10000_images_part_2"
            ]
            for f in dossier.glob("*.jpg")
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = int(row["label"])

        # Chargement de l'image
        image_path = self.chemins.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Image introuvable pour image_id : {image_id}")

        image = Image.open(image_path).convert("RGB")

        # Application des transformations si elles sont définies
        if self.transform:
            image = self.transform(image)

        return image, label