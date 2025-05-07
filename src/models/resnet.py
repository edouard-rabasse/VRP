# resnet_scoring_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from tqdm import tqdm
import cv2               # still needed for Grad-CAM visualisation

# ---------------------------------------------------------------------
# 1) Classifier – ResNet backbone
# ---------------------------------------------------------------------
class ResNetScoringModel(nn.Module):
    """
    ResNet-18 adapted for small 84×84 grey-scale inputs and 2-class output.
    """
    def __init__(self,
                 num_classes: int = 2,
                 input_channels: int = 1,
                 pretrained: bool = False,
                 *,
                 kernel_size: int = 7, freeze=True):
        super().__init__()

        # ❶ Load a vanilla ResNet-18
        self.backbone = models.resnet18(
            weights="IMAGENET1K_V1" if pretrained else None)

        # ❷ Replace the first conv to accept 1-channel images
        #     • shrink kernel to 3×3 and stride to 1
        #     • remove the first max-pool so that 84×84 does not collapse
        self.backbone.conv1 = nn.Conv2d(input_channels, 64,
                                        kernel_size=kernel_size, stride=1,
                                        padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()

        # ❸ Replace the classifier head
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features,
                                     num_classes)

        # ❹ Kaiming initialisation for the new layers
        nn.init.kaiming_normal_(self.backbone.conv1.weight,
                                nonlinearity='relu')
        nn.init.kaiming_normal_(self.backbone.fc.weight,
                                nonlinearity='linear')
        ## freeze all except classifier
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        # for param in self.backbone.fc.parameters():
        #     param.requires_grad = True

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    # 2. Débloquer conv1 + BN1
        for p in self.backbone.conv1.parameters():
            p.requires_grad = True
        for p in self.backbone.bn1.parameters():
            p.requires_grad = True

        # 3. Débloquer layer3, layer4 et la tête
        for name in ["layer3", "layer4", "fc"]:
            for p in getattr(self.backbone, name).parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------
# 2) Training / evaluation helpers (unchanged)
# ---------------------------------------------------------------------
def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets, _ in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

    return running_loss / total, correct / total


def train_model(model,
                train_loader,
                test_loader,
                num_epochs: int = 10,
                device: str = 'cpu',
                learning_rate: float = 5e-4,
                criterion: nn.Module | None = None,
                cfg = None):

    model.to(device)
    print(f"Training on {device} for {num_epochs} epochs — LR={learning_rate}")
    # params + LR différenciés (facultatif mais utile)

    if cfg.model.freeze:
        head_params   = list(model.backbone.fc.parameters()) + \
                        list(model.backbone.layer4.parameters())
        backbone_new  = list(model.backbone.conv1.parameters()) + \
                        list(model.backbone.bn1.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": head_params,  "lr": learning_rate},   # fine-tune rapide
                {"params": backbone_new, "lr": learning_rate/5},   # plus lent
            ],
            weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4
        )
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    criterion  = criterion or nn.CrossEntropyLoss()
    # collect per-epoch metrics
    metrics     = []

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets, _ in tqdm(train_loader,
                                       desc=f"Epoch {epoch+1}/{num_epochs}",
                                       leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds  = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        # compute metrics
        train_loss, train_acc = running_loss / total, correct / total
        # evaluate on test set
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, Train acc={train_acc*100:.2f}%, Test loss={val_loss:.4f}, Test acc={val_acc*100:.2f}%")
        metrics.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    # final return of metrics
    return metrics


# ---------------------------------------------------------------------
# 3) Grad-CAM utility (identical, but we point it to layer4[-1])
# ---------------------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model         = model
        self.target_layer  = target_layer
        self.activations   = None
        self.gradients     = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx):
        self.model.zero_grad()
        out  = self.model(input_tensor)
        loss = out[:, class_idx].sum()
        loss.backward()

        weights     = self.gradients.mean(dim=[0, 2, 3])   # GAP
        act         = self.activations[0] * weights[:, None, None]
        heatmap     = act.sum(dim=0).cpu().numpy()
        heatmap     = np.maximum(heatmap, 0)
        return heatmap / (heatmap.max() + 1e-8)


# ---------------------------------------------------------------------
# 4) Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    cfg = 7


    train_original_path = "data/train/configuration1"
    test_original_path = "data/test/configuration1"
    train_modified_path = f"data/train/configuration{cfg}"
    test_modified_path = f"data/test/configuration{cfg}"
    train_mask_path = f"data/mask_removed/train/mask{cfg}"
    test_mask_path = f"data/mask_removed/test/mask{cfg}"
    image_size = (224,224)

    ## Define the transform to resize the images and convert them to tensors


    from src.data_loader_mask import load_data_train_test
    from src.transform import image_transform_train, image_transform_test, mask_transform


    print("Loading data...")
    train_loader, test_loader = load_data_train_test(
        train_original_path=train_original_path,
        test_original_path=test_original_path,
        train_modified_path=train_modified_path,
        test_modified_path=test_modified_path,
        mask_path_train=train_mask_path,
        mask_path_test=test_mask_path,
        image_transform_train=image_transform_train(size=image_size),
        image_transform_test=image_transform_test(size=image_size),
        mask_transform_train=mask_transform(size=image_size),
        mask_transform_test=mask_transform(size=image_size),
        batch_size=8,
        num_workers=0
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ResNetScoringModel(pretrained=True, input_channels=3)          # or True
    print(model)


    # Grad-CAM: take the last conv block of layer4
    grad_cam = GradCAM(model,
                       model.backbone.layer4[-1].conv2)

    train_model(model, train_loader, test_loader,
                num_epochs=40, device=device, learning_rate=3e-3)
