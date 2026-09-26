import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def get_class_weights(df, num_classes=7):
    """
    Calcule les poids de classes inversement proportionnels à leur fréquence.
    À utiliser dans la loss pour gérer le déséquilibre.
    """
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    weights = total / (num_classes * counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def save_model(model, path, config=None):
    """
    Sauvegarde le modèle et sa config.
    config : dict optionnel (ex: {"fold": 0, "epoch": 10, "val_f1": 0.82})
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config or {}
    }, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(model, path, device="cpu"):
    """
    Recharge les poids dans un modèle existant.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    config = checkpoint.get("config", {})
    print(f"Modèle chargé : {path} | config : {config}")
    return model, config


def compute_metrics(y_true, y_pred):
    """
    Calcule accuracy, balanced accuracy et macro F1.
    y_true, y_pred : listes ou arrays numpy d'entiers
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }