import torch
from .utils import compute_metrics


def train_one_epoch(model, loader, *, optimizer, criterion, device):
    """
    Entraîne le modèle sur un epoch complet.
    Retourne la loss moyenne sur l'epoch.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    """
    Passe le modèle en mode évaluation et retourne (logits, labels)
    en tenseurs CPU, dans l'ordre du dataset (loader sans mélange).
    """
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in loader:
        all_logits.append(model(images.to(device)).cpu())
        all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def evaluate(model, loader, criterion, device):
    """
    Évalue le modèle sans gradient.
    Retourne un dict avec loss, accuracy, balanced_accuracy, f1
    et les logits / labels (tenseurs CPU) pour la calibration.
    """
    logits, labels = predict(model, loader, device)

    metrics = compute_metrics(labels.numpy(), logits.argmax(dim=1).numpy())
    metrics["loss"] = criterion(logits.to(device), labels.to(device)).item()
    metrics["logits"] = logits
    metrics["labels"] = labels

    return metrics
