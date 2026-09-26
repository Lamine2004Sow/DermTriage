import torch
from .utils import compute_metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
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


def evaluate(model, loader, criterion, device):
    """
    Évalue le modèle sans gradient.
    Retourne un dict avec loss, accuracy, balanced_accuracy, f1.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics