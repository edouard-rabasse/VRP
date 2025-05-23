from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert numpy array (from cv2) to PIL Image
        transforms.Resize((84, 84)),  # Resize both image and mask to 84x84
        transforms.ToTensor(),  # Convert to tensor (scales to [0,1])
    ]
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.vgg import VGG16_BN_Weights


class SegmentVGG(nn.Module):
    def __init__(
        self,  # e.g., original vs. modified
        seg_out_channels=1,  # e.g., 1 for binary segmentation
        pretrained=True,
        input_size=(224, 224),
        mask_shape=(10, 10),  # Input image size for upsampling the segmentation,
        freeze=True,
    ):
        super(SegmentVGG, self).__init__()

        # 1. Load pretrained VGG16 with batch normalization
        vgg = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)

        # 2. Keep only the 'features' part of VGG as your encoder (shared feature extractor)
        self.features = (
            vgg.features
        )  # [N, 512, H_out, W_out] after the last conv + pooling

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        # 3. Keep the avgpool from VGG (this is a nn.AdaptiveAvgPool2d((7,7)) by default)
        self.avgpool = vgg.avgpool

        # 4. Create a new classifier head
        #    - Typically VGG16 has [Linear(512*7*7, 4096), ReLU, Dropout, Linear(4096, 4096), ReLU, Dropout, Linear(4096, 1000)]

        # 5. Create a segmentation decoder/head
        #    - We end up with a 512-channel feature map after `self.features`.
        #    - We upsample it back to 'input_size' (e.g., 224x224) or smaller if you prefer.
        #    - Below is a minimal example: conv -> upsample -> conv -> upsample -> final conv
        #    - You can add more layers, skip-connections, etc., to improve segmentation quality.
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # At this point, if the input was 224x224,
            # after VGG, the spatial size is 7x7 before avgpool.
            # We did 2x + 2x upsampling => 28x28
            # Another upsampling might be needed:
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=mask_shape, mode="bilinear", align_corners=False),
            # Final conv to get the correct number of segmentation channels (1 for binary)
            nn.Conv2d(64, seg_out_channels, kernel_size=3, padding=1),
        )

        # 6. (Optional) Initialize new layers
        #    The pretrained VGG layers are already initialized.
        #    For newly added layers, you can do e.g. Kaiming or Xavier init.
        for m in self.segmentation_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Shared feature extraction
        features = self.features(x)  # shape: [N, 512, H_out, W_out]

        # Classification head: Global avgpool + linear
        pooled = self.avgpool(features)  # shape: [N, 512, 7, 7]
        flattened = torch.flatten(pooled, 1)  # shape: [N, 512*7*7]

        # Segmentation head
        seg_logits = self.segmentation_head(
            features
        )  # shape: [N, 1, input_size[0], input_size[1]]

        return seg_logits


def train_model_multi_task(
    model, train_loader, test_loader, *, num_epochs, device, learning_rate, cfg=None
):
    """
    Train the multi-task model.
    Args:
        model: the multi-task model.
        train_loader: DataLoader for training.
        test_loader: DataLoader for testing (evaluation).
        num_epochs: Number of epochs to train.
        device: 'cpu' or 'cuda'.
        learning_rate: learning rate for optimizer.
        lambda_seg: weight for the segmentation loss.
    Returns:
        model, optimizer, scheduler after training.
    """
    model.to(device)
    model.train()
    metrics = []  # list to collect metrics per epoch

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    criterion_seg = nn.BCEWithLogitsLoss(
        reduction="none"
    )  # for pixel-wise mask prediction

    for epoch in range(num_epochs):
        model.train()
        running_seg_loss = 0.0
        total = 0

        for images, labels, masks in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        ):
            images = images.to(device)  # shape: [N, 3, H, W] (e.g., 84x84)
            labels = labels.to(device)  # shape: [N]
            masks = masks.to(device).float()  # shape: [N, 1, H, W]

            optimizer.zero_grad()

            feats = model.features(images)  # shape: [N, 512, H_out, W_out]
            seg_logits = model.segmentation_head(feats)
            loss_seg_map = criterion_seg(seg_logits, masks)

            # vecteur booléen : 1 si l'image a au moins un pixel positif
            mask_present = (
                masks.view(masks.size(0), -1).sum(dim=1) > 0
            ).float()  # shape (N,)
            num_pos = mask_present.sum().item()

            # on pèse chaque carte de perte par mask_present
            loss_seg = (
                loss_seg_map.view(loss_seg_map.size(0), -1).mean(dim=1) * mask_present
            ).sum() / (
                mask_present.sum() + 1e-6
            )  # moyenne sur les images « valides »

            (loss_seg).backward()
            optimizer.step()
            running_seg_loss += loss_seg.item() * num_pos
            total += num_pos

        scheduler.step()
        epoch_seg_loss = running_seg_loss / total

        # evaluate on test set
        test_seg_loss = 0.0
        total = 0
        model.eval()
        for images, labels, masks in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device).float()

            with torch.no_grad():
                seg_logits = model(images)

                loss_seg_map = criterion_seg(seg_logits, masks)

                # vecteur booléen : 1 si l'image a au moins un pixel positif
                mask_present = (
                    masks.view(masks.size(0), -1).sum(dim=1) > 0
                ).float()  # shape (N,)
                num_pos = mask_present.sum().item()

                # on pèse chaque carte de perte par mask_present
                loss_seg = (
                    loss_seg_map.view(loss_seg_map.size(0), -1).mean(dim=1)
                    * mask_present
                ).sum() / (mask_present.sum() + 1e-6)
            test_seg_loss += loss_seg.item() * num_pos
            total += num_pos
        # Record test metrics
        metrics.append(
            {
                "epoch": epoch + 1,
                "seg_loss": epoch_seg_loss,
                "test_seg_loss": test_seg_loss / total,
            }
        )

    return metrics


if __name__ == "__main__":
    # Assuming your directory paths are defined:
    original_path, modified_path, mask_path = (
        "MSH/MSH/plots/configuration1",
        "MSH/MSH/plots/configuration3",
        "data/MSH/mask3",
    )

    train_original_path = "data/train/configuration1"
    test_original_path = "data/test/configuration1"
    train_modified_path = "data/train/configuration3"
    test_modified_path = "data/test/configuration3"
    train_mask_path = "data/mask_removed/configuration3"
    test_mask_path = "data/mask_removed/configuration3"
    image_size = (224, 224)

    ## Define the transform to resize the images and convert them to tensors

    from src.data_loader_mask import load_data_train_test
    from src.transform import (
        image_transform_train,
        image_transform_test,
        mask_transform,
    )

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
        batch_size=1,
        num_workers=0,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SegmentVGG()
    model.to(device)

    # Train for, say, 10 epochs and use a lambda weight of 1.0 for segmentation loss.
    model, optimizer, scheduler = train_model_multi_task(
        model,
        train_loader,
        test_loader,
        num_epochs=50,
        device=device,
        learning_rate=1e-3,
        lambda_seg=0.1,
    )
    # print results
    print("Training completed.")

    # test the model on test set and save an image of the overlay
