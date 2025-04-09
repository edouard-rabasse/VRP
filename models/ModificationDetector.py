import torch.nn as nn
from torchvision import models

class ModificationDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared encoder (ResNet18)
        self.encoder = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # Remove last layers
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)  # Binary classification
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()  # Mask probability
        )

    def forward(self, x):
        features = self.encoder(x)  # Shared features
        classification = self.cls_head(features)
        segmentation = self.seg_head(features)
        return classification, segmentation