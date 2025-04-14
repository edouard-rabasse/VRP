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
    def __init__(self, samples, transform=None):
        """
        Args:
            samples (list): A list of tuples (img_path, label, mask_path).
            transform (callable, optional): Transform to apply to the images and masks.
        """
        self.all_samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.all_samples[idx]
        
        # Load the image using cv2
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load or create mask
        if label == 0 or mask_path is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

        # Apply the transform if provided.
        # Note: Make sure the transform can work on numpy images,
        # especially if one of the transforms is ToPILImage().
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            # Ensure image becomes CxHxW (if image is HxWx3)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

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


def load_data_mask(
    original_path,
    modified_path,
    batch_size=32,
    transform_train=None,
    transform_test=None,
    train_ratio=0.8,
    image_size=(284, 284),
    num_workers=4,
    mask_path=None,
    num_max=None
):
    # Build the master sample list
    original_images = [os.path.join(original_path, f)
                       for f in os.listdir(original_path)
                       if f.endswith(('.jpg', '.png'))]
    
    modified_images = [os.path.join(modified_path, f)
                       for f in os.listdir(modified_path)
                       if f.endswith(('.jpg', '.png'))]
    
    if mask_path is not None:
        all_samples = (
            [(path, 0, None) for path in original_images] +
            [(path, 1, os.path.join(mask_path, os.path.basename(path)))
             for path in modified_images]
        )
    else:
        all_samples = (
            [(path, 0, None) for path in original_images] +
            [(path, 1, None) for path in modified_images]
        )
    
    # If you want to restrict to a maximum number of samples
    if num_max is not None:
        all_samples = all_samples[:num_max]
    
    # Shuffle the samples
    np.random.shuffle(all_samples)
    
    # Split the sample list according to train_ratio
    train_size = int(train_ratio * len(all_samples))
    train_samples = all_samples[:train_size]
    test_samples  = all_samples[train_size:]
    
    # Create separate dataset instances with different transforms
    train_dataset = CustomDataset(train_samples, transform=transform_train)
    test_dataset = CustomDataset(test_samples, transform=transform_test)
    
    # Build DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
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
    
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((284, 284)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # Normalize with ImageNet's mean and std if using a pretrained model like VGG16
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform for testing (deterministic)
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((284, 284)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_loader, test_loader = load_data_mask(
        original_path=original_path,
        modified_path=modified_path,
        batch_size=32,
        transform_train=transform_train,
        transform_test=transform_test,
        train_ratio=0.8,
        image_size=(284, 284),
        num_workers=4,
        mask_path=mask_path,
        num_max=None
    )

    for samples in train_loader:
        print("sample size", samples.shape)
        print(vars(samples))
    
        # print("singular mask shape:", masks[index].shape)
        # cv2.imwrite("output/mask.png", masks[index].permute(1,2,0).numpy()*255)
        # cv2.imwrite("output/image.png", images[index].permute(1, 2, 0).numpy()*255)
        # break  # Just to print one instance