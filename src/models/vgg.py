from torchvision import models

import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm


def load_vgg():
    model = models.vgg16(weights='DEFAULT')
    model.classifier[6] = nn.Linear(4096, 2)  # Change the last layer to have 2 classes
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return model



def precompute_model(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    all_outputs = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for inputs, labels, masks in dataloader:
            inputs = inputs.to(device)
            outputs = model.features(inputs)
            outputs = model.avgpool(outputs)
            all_outputs.append(outputs.cpu().detach())
            all_labels.append(labels.cpu())
            all_masks.append(masks.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0)


    output_dataset = torch.utils.data.TensorDataset(all_outputs, all_labels, all_masks)

    return output_dataset


    

def train_vgg(model, train_loader, test_loader,*,device='cpu', num_epochs=20, learning_rate=0.001, criterion=None,cfg=None, grad_layer=5):
    import torch.optim as optim
    # Send model to device
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.classifier[grad_layer:].parameters():
        param.requires_grad = True

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    results = [f"Parameters: {num_epochs} epochs, {learning_rate} learning rate"]

    # Lists to track metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    # Training loop
    for epoch in range(num_epochs):
        # === Training Phase ===
        model.classifier.train()
        running_loss, correct_preds, total = 0.0, 0, 0

        for input, labels, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            input, labels = input.to(device), labels.to(device)
            # Select the CLS token (first token) for classification
            # cls_features = torch.flatten(input, start_dim=1)

            optimizer.zero_grad()
            outputs = model(input)
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
        model.classifier.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for input, labels, mask in test_loader:
                input, labels = input.to(device), labels.to(device)
                outputs = model(input)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                preds = torch.argmax(outputs, dim=1)
                # check that pred and labels are in {0,1}
                if torch.any(preds > 1) or torch.any(preds < 0):
                    print(f"Predictions out of range: {preds}")
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
        results.append(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
    # reactivate gradients
    for param in model.parameters():
        param.requires_grad = True
    return results
