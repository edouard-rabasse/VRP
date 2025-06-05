import torch.nn as nn
import torch.nn.functional as F

import torch
from tqdm import tqdm

from torch.utils.data import TensorDataset


from timm.scheduler import CosineLRScheduler


def load_deit(model_name, device, num_classes=2, freeze=True):
    if model_name == "deit_tiny":
        # Load the pretrained DEIT model from Facebook Research
        # This will download the model if not already cached
        # torch.hub.clear()
        model = torch.hub.load(
            "facebookresearch/deit:main", "deit_tiny_patch16_224", pretrained=True
        )

        # Changing the last layer to have 2 classes
        in_features = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_features, num_classes))
        # nn.init.xavier_uniform_(model.head.weight); nn.init.zeros_(model.head.bias)

        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
    else:
        raise ValueError("Unknown model name: {}".format(model_name))
    print(model.head)
    return model


def precompute_deit_tiny_features(model, dataloader, device="cpu"):
    model.eval()
    model.to(device)
    list_outputs = []
    list_labels = []
    list_masks = []

    with torch.no_grad():
        try:
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                inputs = model.forward_features(inputs)
                list_outputs.append(inputs.cpu())
                list_labels.append(labels.cpu())
            outputs = torch.cat(list_outputs, dim=0)
            labels = torch.cat(list_labels, dim=0)

            return TensorDataset(outputs, labels)

        except Exception as e:  # depends on the type of dataloader
            for inputs, labels, masks in dataloader:
                inputs = inputs.to(device)
                inputs = model.forward_features(inputs)
                list_outputs.append(inputs)
                list_labels.append(labels)
                list_masks.append(masks)

            outputs = torch.cat(list_outputs, dim=0)
            labels = torch.cat(list_labels, dim=0)
            masks = torch.cat(list_masks, dim=0)

            return TensorDataset(outputs, labels, masks)


def train_deit(
    model,
    train_loader,
    test_loader,
    device="cpu",
    num_epochs=20,
    learning_rate=0.001,
    criterion=None,
):
    import torch.optim as optim

    # Send model to device
    model.to(device)
    model.eval()  #

    model.train()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )

    # Initialize metrics list
    metrics = []
    # Training loop
    for epoch in range(num_epochs):
        # === Training Phase ===
        running_loss, correct_preds, total = 0.0, 0, 0

        for inputs, labels, mask in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            # Select the CLS token (first token) for classification
            cls_features = inputs[:, 0, :]

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

        # === Validation Phase ===
        model.head.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for inputs, labels, mask in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                cls_features = inputs[:, 0, :]
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

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )
        metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    return metrics


def train_deit_no_precompute(
    model,
    train_loader,
    test_loader,
    *,
    device="cpu",
    num_epochs=20,
    learning_rate=0.001,
    criterion=None,
    cfg=None,
):
    import torch.optim as optim

    # Send model to device
    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=learning_rate)
    # for p in model.blocks[-1].parameters(): p.requires_grad_(True)
    # head_params  = [p for n,p in model.named_parameters() if n.startswith('head')]
    # block_params = list(model.blocks[-1].parameters())
    # optimizer = optim.AdamW([
    #     {'params': head_params,  'lr': 3e-4},
    #     {'params': block_params,'lr': 1e-5}
    # ], weight_decay=1e-4)

    num_steps = num_epochs * len(train_loader)
    warm_steps = int(0.1 * num_steps)

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps - warm_steps,
        lr_min=1e-6,
        warmup_lr_init=1e-6,
        warmup_t=warm_steps,
    )

    # Initialize metrics list
    metrics = []
    # Training loop
    for epoch in range(num_epochs):
        # === Training Phase ===
        running_loss, correct_preds, total = 0.0, 0, 0
        i = 0

        for inputs, labels, mask in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        ):
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)
            # Select the CLS token (first token) for classification

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(i)
            i += 1

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_preds += torch.sum(preds == labels).item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct_preds / total

        # === Validation Phase ===
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for inputs, labels, masks in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                # print(preds, labels)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)

                all_val_preds.append(preds.cpu())
                all_val_labels.append(labels.cpu())

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )

        # Record epoch metrics
        metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    return metrics


def train_deit_mask(
    model,
    train_loader,
    test_loader,
    *,
    device="cpu",
    num_epochs=20,
    learning_rate=0.001,
    criterion=None,
):
    import torch.optim as optim

    # Send model to device
    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=learning_rate)

    # Initialize metrics list
    metrics_mask = []
    # Training loop
    for epoch in range(num_epochs):
        # === Training Phase ===
        model.head.train()
        running_loss, correct_preds, total = 0.0, 0, 0

        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            # Select the CLS token (first token) for classification
            cls_features = inputs[:, 0, :]

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

        # === Validation Phase ===
        model.head.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                cls_features = inputs[:, 0, :]
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

        print(
            f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )
        metrics_mask.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    return metrics_mask
