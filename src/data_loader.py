from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from src.preprocessing import get_transform
import torch
from torch.utils.data import TensorDataset



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
    full_dataset = datasets.ImageFolder(root=data_path, transform=transform,)
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

def load_data(data_path, batch_size=32, transform=None,train_ratio=0.8, image_size=(284, 284),num_workers=4, num_max = 2):
    """Load the dataset and return the DataLoader for training and testing.
    Args:
        data_path (str): Path to the dataset.
        batch_size (int): Batch size for the DataLoader.
        train_ratio (float): Ratio of the dataset to use for training.
        image_size (tuple): Size to resize the images to.
    Returns:
        DataLoader, DataLoader: The training and testing DataLoaders.
    """
    if transform is None:
        transform = get_transform(image_size=image_size)
    
    dataset = get_dataset(data_path, transform, image_size=image_size)
    if num_max is not None:
        dataset.samples = dataset.samples[:num_max]
        dataset.targets = dataset.targets[:num_max]
        dataset.imgs = dataset.imgs[:num_max]
    

    train_dataset, test_dataset = split_dataset(dataset, train_ratio=train_ratio)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def precompute_deit(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)
    list_outputs = []
    list_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            features = model.forward_features(inputs)
            list_outputs.append(features)
            list_labels.append(labels)

    outputs = torch.cat(list_outputs, dim=0)
    labels = torch.cat(list_labels, dim=0)

    return TensorDataset(outputs, labels)    