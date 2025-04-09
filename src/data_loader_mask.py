from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset
from .preprocessing import get_transform
import torch
from torch.utils.data import TensorDataset
import os
import cv2
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, original_dir, modified_dir, mask_dir=None, transform=None):
        self.transform = transform
        
        # Collect original and modified image paths
        self.original_images = [os.path.join(original_dir, f) 
                               for f in os.listdir(original_dir) 
                               if f.endswith(('.jpg', '.png'))]
                               
        self.modified_images = [os.path.join(modified_dir, f) 
                              for f in os.listdir(modified_dir) 
                              if f.endswith(('.jpg', '.png'))]
        
        # Combine all samples (originals + modifieds)
        if mask_dir is not None:
            self.all_samples = (
                [(path, 0, None) for path in self.original_images] +  # 0=original, no mask
                [(path, 1, os.path.join(mask_dir, os.path.basename(path))) 
                for path in self.modified_images]  # 1=modified, with mask
            )
        else:
            self.all_samples = (
                [(path, 0, None) for path in self.original_images] +  # 0=original, no mask
                [(path, 1, None) for path in self.modified_images]  # 1=modified, no mask
            )

    def __len__(self):
        return len(self.all_samples)
        
    def __getitem__(self, idx):
        img_path, label, mask_path = self.all_samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        # Load mask (zeros for originals or if no mask path)
        if label == 0 or mask_path is None:
            mask = np.zeros((image.shape[0],image.shape[1]), dtype=np.float64)
   

        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/ 255.0
  


        

        
        # use transforms
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:   
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).float()

        
        return image, torch.tensor(label, dtype=torch.long), mask


def get_dataset(data_path, transform, image_size=(284, 284)):
    """Return the dataset with the specified transformations.
    Args:
        data_path (str): Path to the dataset.
        transform (callable): Transform to be applied to the images.
        image_size (tuple): Size to resize the images to.
    Returns:
        Dataset: The dataset with the specified transformations.
    """
    input_shape = (3, image_size[0], image_size[1])
    full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return full_dataset


def split_dataset(dataset, train_ratio=0.8):
    """Split the dataset into training and testing sets.
    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): Ratio of the dataset to use for training.
    Returns:
        Dataset, Dataset: The training and testing datasets.
    """
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """Return the DataLoader for the dataset.
    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers for loading data.
    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def load_data_mask(original_path, modified_path, batch_size=32, transform=None, train_ratio=0.8, 
              image_size=(284, 284), num_workers=4, mask_path=None):
    """Load the dataset and return the DataLoader for training and testing.
    Args:
        original_path (str): Path to the original images.
        modified_path (str): Path to the modified images.
        batch_size (int): Batch size for the DataLoader.
        transform (callable): Transform to be applied to the images.
        train_ratio (float): Ratio of the dataset to use for training.
        image_size (tuple): Size to resize the images to.
        num_workers (int): Number of workers for loading data.
        mask_path (str): Path to the mask images.
    Returns:
        DataLoader, DataLoader: The training and testing DataLoaders.
    """
    if transform is None:
        transform = get_transform(image_size=image_size)
    
    dataset = CustomDataset(original_path, modified_path, mask_path, transform=transform)
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=train_ratio)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def precompute_deit(model, dataloader, device='cpu'):
    """Precompute features using a DeiT model.
    Args:
        model: The DeiT model.
        dataloader: DataLoader for the dataset.
        device: Device to use ('cpu' or 'cuda').
    Returns:
        TensorDataset: Dataset with precomputed features and labels.
    """
    model.eval()
    model.to(device)
    list_outputs = []
    list_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model.forward_features(inputs)
            list_outputs.append(features.cpu())
            list_labels.append(labels)

    outputs = torch.cat(list_outputs, dim=0)
    labels = torch.cat(list_labels, dim=0)

    return TensorDataset(outputs, labels)


if __name__ == "__main__":
    # Example usage
    original_path = 'MSH/MSH/plots/configuration3'
    modified_path = 'MSH/MSH/plots/configuration5'
    mask_path = 'data/MSH/mask'
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((284, 284)),
        transforms.ToTensor(),
    ])
    
    train_loader, test_loader = load_data_mask(
        original_path=original_path,
        modified_path=modified_path,
        batch_size=32,
        transform=transform,
        train_ratio=0.8,
        image_size=(284, 284),
        num_workers=4,
        mask_path=mask_path
    )
    dataset = CustomDataset(original_path, modified_path, mask_path, transform=transform)
    print(type(dataset[0]))
    print(dataset[0][0].shape)  # Image shape
    for images, labels, masks in train_loader:
        print("Images shape:", images.shape)
        print("Labels shape:", labels.shape)
        print("Masks shape:", masks.shape)
        break  # Just to print one instance