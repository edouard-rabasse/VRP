import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, original_dir=None, modified_dir=None, mask_dir=None,
                 image_transform=None, mask_transform=None, samples=None):
        """
        Args:
            original_dir (str): Path to original images.
            modified_dir (str): Path to modified images.
            mask_dir (str): Path to mask images.
            image_transform (callable): Transform to be applied to images.
            mask_transform (callable): Transform to be applied to masks.
            samples (list, optional): Pre-built sample list, each a tuple (img_path, label, mask_path).
        """
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        if samples is not None:
            self.all_samples = samples
        else:
            self.original_images = [os.path.join(original_dir, f)
                                    for f in os.listdir(original_dir)
                                    if f.lower().endswith(('.jpg', '.png'))]
            self.modified_images = [os.path.join(modified_dir, f)
                                    for f in os.listdir(modified_dir)
                                    if f.lower().endswith(('.jpg', '.png'))]
            if mask_dir is not None:
                self.all_samples = ([(path, 0, None) for path in self.original_images] +
                                    [(path, 1, os.path.join(mask_dir, os.path.basename(path)))
                                     for path in self.modified_images])
            else:
                self.all_samples = ([(path, 0, None) for path in self.original_images] +
                                    [(path, 1, None) for path in self.modified_images])
        self.imgs = self.all_samples  # for backward compatibility

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.all_samples[idx]
        
        # Load image using cv2 and convert to RGB.
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Unable to load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask: if none provided or label indicates original, create a mask of zeros.
        if label == 0 or mask_path is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Apply image transform (expects a NumPy array)
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Apply mask transform: here we use a simple transform without normalization that expects a single channel.
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, torch.tensor(label, dtype=torch.long), mask

def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def load_data_mask(
    original_path,
    modified_path,
    batch_size=32,
    image_transform_train=None,
    image_transform_test=None,
    mask_transform_train=None,
    mask_transform_test=None,
    train_ratio=0.8,
    image_size=(224, 224),
    num_workers=4,
    mask_path=None,
    num_max=None
):
    """
    Load the dataset and return separate training and testing DataLoaders using separate transforms.
    """
    # Create a base dataset to get the sample list (without any transform)
    base_dataset = CustomDataset(
        original_dir=original_path,
        modified_dir=modified_path,
        mask_dir=mask_path,
        image_transform=None,
        mask_transform=None
    )
    if num_max is not None:
        base_dataset.all_samples = base_dataset.all_samples[:num_max]
    
    all_samples = base_dataset.all_samples
    num_samples = len(all_samples)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    train_size = int(train_ratio * num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Build separate sample lists for train and test.
    train_samples = [all_samples[i] for i in train_indices]
    test_samples = [all_samples[i] for i in test_indices]
    
    # Create dataset instances with appropriate transforms.
    train_dataset = CustomDataset(
        samples=train_samples,
        image_transform=image_transform_train,
        mask_transform=mask_transform_train
    )
    test_dataset = CustomDataset(
        samples=test_samples,
        image_transform=image_transform_test,
        mask_transform=mask_transform_test
    )
    
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

# Example usage:

if __name__ == "__main__":
    original_path = 'MSH/MSH/plots/configuration3'
    modified_path = 'MSH/MSH/plots/configuration5'
    mask_path = 'data/MSH/mask'
    
    # Define transforms for images and masks.
    image_transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image_transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # For the mask, we usually only need resizing and conversion to tensor.
    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # no normalization, retains a single channel
    ])
    
    train_loader, test_loader = load_data_mask(
        original_path=original_path,
        modified_path=modified_path,
        batch_size=32,
        image_transform_train=image_transform_train,
        image_transform_test=image_transform_test,
        mask_transform_train=mask_transform,
        mask_transform_test=mask_transform,
        train_ratio=0.8,
        image_size=(224, 224),
        num_workers=4,
        mask_path=mask_path
    )
    
    # Test one batch to check outputs.
    for images, labels, masks in train_loader:
        print("Images shape:", images.shape)  # Expected: [batch, 3, 224, 224]
        print("Labels shape:", labels.shape)
        print("Masks shape:", masks.shape)      # Expected: [batch, 1, 224, 224]
        # Optionally, save one image and mask.
        cv2.imwrite("output/image.png", images[0].permute(1, 2, 0).numpy() * 255)
        cv2.imwrite("output/mask.png", masks[0].permute(1, 2, 0).numpy() * 255)
        break
