import torch
import torch.nn as nn
from models.ModificationDetector import ModificationDetector
from src.ModificationDataset import ModificationDataset
from torchvision import transforms
import os
from tqdm import tqdm



def combined_loss(cls_pred, seg_pred, cls_target, seg_target):
    # Classification loss (binary cross-entropy)
    cls_loss = nn.BCEWithLogitsLoss()(cls_pred.squeeze(), cls_target.float())
    
    # Segmentation loss (only applied to modified images)
    seg_loss = nn.BCELoss()(seg_pred.squeeze(), seg_target.squeeze())
    
    return cls_loss + seg_loss  # Total loss


from torch.utils.data import DataLoader

# Define transformations
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])

# Initialize dataset and loader
dataset = ModificationDataset(
    original_dir="MSH/MSH/plots/configuration3",
    modified_dir="MSH/MSH/plots/configuration5",
    mask_dir="data/MSH/mask"
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model and optimizer
model = ModificationDetector()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(loader.dataset[0][0].shape)
# Training loop
for epoch in range(10):
    print(torch.cuda.memory_summary())
    for images, labels, masks in tqdm(loader, desc=f"Epoch {epoch+1}/10", leave=True):
        print("before images")
        print(torch.cuda.memory_summary())
        images = images.to(device)
        print("before labels")
        print(torch.cuda.memory_summary())
        labels = labels.to(device)
        print("before masks")
        print(torch.cuda.memory_summary())
        masks = masks.to(device)
        print(torch.cuda.memory_summary())

        optimizer.zero_grad()
        
        # Forward pass
        cls_pred, seg_pred = model(images)
        print("after forward pass")
        
        # Compute loss
        loss = combined_loss(cls_pred, seg_pred, labels, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()