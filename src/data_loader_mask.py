import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
from PIL import Image
import random
import torchvision.transforms.functional as TF

class CustomDataset(Dataset):
    def __init__(self, original_dir=None, modified_dir=None, mask_dir=None,
                 image_transform=None, mask_transform=None, samples=None, augment=False):
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
                self.all_samples = ([(path, 0, None) for path in self.modified_images] +
                                    [(path, 1, os.path.join(mask_dir, os.path.basename(path)))
                                     for path in self.original_images])
            else:
                self.all_samples = ([(path, 0, None) for path in self.modified_images] +
                                    [(path, 1, None) for path in self.original_images])
        self.imgs = self.all_samples  # for backward compatibility
        self.augment = augment

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.all_samples[idx]
        
        # Load image using cv2 and convert to RGB.
        img = Image.open(img_path).convert("RGB")
        if img is None:
            raise ValueError(f"Unable to load image at {img_path}")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask: if none provided or label indicates original, create a mask of zeros.
        if label == 0 or mask_path is None:
            mask = np.zeros((img.size[0], img.size[1]), dtype=np.uint8)
            mask = Image.new("L", img.size, 0)
        else:
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale


        if self.augment:
        # random transforms
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            # img = TF.rotate(img, angle)
            # mask = TF.rotate(mask, angle)

        # Apply image transform (expects a PIL array)
        if self.image_transform:
            image = self.image_transform(img)
        else:
            # to tensor
            image = transforms.ToTensor()(img)
        del img  # Free memory

        # Apply mask transform: here we use a simple transform without normalization that expects a single channel.
        if self.mask_transform:
            mask_tensor = self.mask_transform(mask)
        else:
            mask_tensor = transforms.ToTensor()(mask)
        del mask  # Free memory

        # check if mask_tensor is None
        if mask_tensor is None:
            raise ValueError(f"Mask tensor is None for image at {img_path}")
        
        return image, torch.tensor(label, dtype=torch.long), mask_tensor

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

def load_data_train_test(train_original_path,
                         test_original_path,
                         train_modified_path,
                         test_modified_path,
                         mask_path_train=None,
                         mask_path_test=None,
                         batch_size=32,
                         image_transform_train=None,
                        image_transform_test=None,
                        mask_transform_train=None,
                        mask_transform_test=None,
                        image_size=(224, 224),
                        num_workers=4,
                        num_max=None
                        ):
    train_set = CustomDataset(
        original_dir=train_original_path,
        modified_dir=train_modified_path,
        mask_dir=mask_path_train,
        image_transform=image_transform_train,
        mask_transform=mask_transform_train,
        augment=True)
    test_set = CustomDataset(
        original_dir=test_original_path,
        modified_dir=test_modified_path,
        mask_dir=mask_path_test,
        image_transform=image_transform_test,
        mask_transform=mask_transform_test,
        augment=False)
    
    train_loader = get_dataloader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = get_dataloader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
    

if __name__ == "__main__":
    train_original_path = 'data/train/configuration1'
    train_modified_path = 'data/train/configuration2'
    test_original_path = 'data/test/configuration1'
    test_modified_path = 'data/test/configuration2'
    train_mask_path = 'data/mask/train/mask2'
    test_mask_path = 'data/mask/test/mask2'
    
    # Define transforms for images and masks.
    image_transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image_transform_test = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # For the mask, we usually only need resizing and conversion to tensor.
    mask_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # no normalization, retains a single channel
    ])
    
    train_loader, test_loader = load_data_train_test(
        train_original_path=train_original_path,
        test_original_path=test_modified_path,
        train_modified_path=test_modified_path,
        test_modified_path=test_modified_path,
        batch_size=32,
        image_transform_train=image_transform_train,
        image_transform_test=image_transform_test,
        mask_transform_train=mask_transform,
        mask_transform_test=mask_transform,
        image_size=(224, 224),
        num_workers=4,
        mask_path_train=train_mask_path,
        mask_path_test=test_mask_path,
    )
    
    # Test one batch to check outputs.
    for images, labels, masks in train_loader:
        print("Images shape:", images.shape)  # Expected: [batch, 3, 224, 224]
        print("Labels shape:", labels.shape)
        print("Masks shape:", masks.shape)      # Expected: [batch, 1, 224, 224]
        # Optionally, save one image and mask.
        
        cv2.imwrite("output/image.png", images[0].permute(1, 2, 0).numpy()[:,:,::-1] * 255)
        cv2.imwrite("output/mask.png", masks[0].permute(1, 2, 0).numpy() * 255)
        break
