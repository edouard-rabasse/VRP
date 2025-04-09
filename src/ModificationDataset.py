import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class ModificationDataset(Dataset):
    def __init__(self, original_dir, modified_dir, mask_dir, transform=None):
        self.transform = transform
        
        # Collect original and modified image paths
        self.original_images = [os.path.join(original_dir, f) 
                               for f in os.listdir(original_dir) 
                               if f.endswith(('.jpg', '.png'))]
                               
        self.modified_images = [os.path.join(modified_dir, f) 
                              for f in os.listdir(modified_dir) 
                              if f.endswith(('.jpg', '.png'))]
        
        # Combine all samples (originals + modifieds)
        self.all_samples = (
            [(path, 0, None) for path in self.original_images] +  # 0=original, no mask
            [(path, 1, os.path.join(mask_dir, os.path.basename(path))) 
             for path in self.modified_images]  # 1=modified, with mask
        )

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.all_samples[idx]
        
        # Load image
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # Load mask (zeros for originals)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32) if label == 0 \
               else cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
        
        # to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, torch.tensor(label), mask.unsqueeze(0)  # Add channel dim