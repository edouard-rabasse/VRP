import torch.nn as nn
import torch.nn.functional as F

import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset
print("[DEBUG] deit_tiny.py loaded")



def load_deit(model_name, device, out_features=2):
    if model_name == 'deit_tiny':
        # Load the pretrained DEIT model from Facebook Research
        # This will download the model if not already cached
        # torch.hub.clear()
        model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)

        # Changing the last layer to have 2 classes
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 2)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))
    return model



def precompute_deit_tiny_features(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)
    list_outputs = []
    list_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model.forward_features(inputs)
            list_outputs.append(features)
            list_labels.append(labels)

    outputs = torch.cat(list_outputs, dim=0)
    labels = torch.cat(list_labels, dim=0)

    return TensorDataset(outputs, labels)

def train_deit(model, train_loader, test_loader,device='cpu', num_epochs=20, learning_rate=0.001, criterion=None):
    import torch.optim as optim
    # Send model to device
    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=learning_rate)

    # Lists to track metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    val_f1_scores = []

    # Training loop
    for epoch in range(num_epochs):
        # === Training Phase ===
        model.head.train()
        running_loss, correct_preds, total = 0.0, 0, 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            # Select the CLS token (first token) for classification
            cls_features = features[:, 0, :]

            optimizer.zero_grad()
            outputs = model.head(cls_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum(preds == labels).item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct_preds / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # === Validation Phase ===
        model.head.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                cls_features = features[:, 0, :]
                outputs = model.head(cls_features)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)

                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Compute the F1-score on the validation set (binary average for 2 classes)
        all_val_preds = torch.cat(all_val_preds).numpy()
        all_val_labels = torch.cat(all_val_labels).numpy()

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")