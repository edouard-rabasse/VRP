import torch
from sklearn.metrics import confusion_matrix
import numpy as np

def get_confusion_matrix(model, test_loader, device="cuda"):
    """
    Calcule la matrice de confusion pour `model` sur `test_loader`.

    Parameters
    ----------
    model : torch.nn.Module
        Réseau déjà chargé et, si besoin, envoyé sur `device`.
    test_loader : torch.utils.data.DataLoader
        DataLoader de validation / test (batch, label, _).
    device : str, optional
        "cuda" ou "cpu" (par défaut "cuda").

    Returns
    -------
    numpy.ndarray
        Matrice de confusion (shape : [n_classes, n_classes]).
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            # la plupart de vos DataLoader renvoient (image, label, mask)
            images, labels = batch[0].to(device), batch[1].to(device)

            outputs = model(images)

            # si le modèle renvoie un tuple / liste → on garde le 1er
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    # concatène les batches en un seul vecteur
    y_pred   = torch.cat(all_preds).numpy()
    y_true   = torch.cat(all_targets).numpy()

    return confusion_matrix(y_true, y_pred)

# --- Exemple d'utilisation ------------------------------------
# cm = get_confusion_matrix(model, test_loader, device="cuda")
# print(cm)
