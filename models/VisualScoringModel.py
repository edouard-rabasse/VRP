import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# print("[DEBUG] Loaded: VisualScoringModel.py")

class VisualScoringModel(nn.Module):
    def __init__(self,input_shape=(1, 84, 84)):
        super(VisualScoringModel, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # The output shape after conv3 is [N, 64, 7, 7]

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = self._forward_conv(dummy_input)
            self.flattened_size = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 2)

        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')

    def _forward_conv(self, x):
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.01)
        return self.fc2(x)


def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets, mask in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

    return running_loss / total, correct / total

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    device: str = 'cpu',
    learning_rate: float = 1e-3,
    criterion: nn.Module = None,
    ):
    # Send model to device
    model.to(device)
    model.train()
    print("Training with parameters:")
    print(f"  - Device: {device}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Number of epochs: {num_epochs}")

    # Optimizer, scheduler, and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc*100:.2f}%")

        # Optional: run evaluation on test set
    eval_loss, eval_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"           Test Loss  = {eval_loss:.4f}, Test Acc  = {eval_acc*100:.2f}%\n")


import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        print("Output shape:", output.shape)
        loss = output[:, class_index].sum()
        loss.backward()

        # Compute the weights
        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])  # global average pooling
        activations = self.activations[0]  # remove batch dimension
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = activations.sum(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8  # normalize
        return heatmap
    

