# data_loader_mask.py : creates a dataset class for loading images and masks, and provides functions to load data into DataLoader objects.

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
    def __init__(
        self,
        original_dir=None,
        modified_dir=None,
        mask_dir=None,
        image_transform=None,
        mask_transform=None,
        samples=None,
        augment=False,
        range=None,
        return_filenames: bool = False,
    ):
        """
        Args:
            original_dir (str): Path to original images.
            modified_dir (str): Path to modified images.
            mask_dir (str): Path to mask images.
            image_transform (callable): Transform to be applied to images.
            mask_transform (callable): Transform to be applied to masks.
            samples (list, optional): Pre-built sample list, each a tuple (img_path, label, mask_path).
            augment (bool): Whether to apply augmentations.
            range (tuple): Range of indices to sample from the dataset.
            return_filenames (bool): Whether to return filenames as 4th argument.
        """
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        if samples is not None:
            self.all_samples = samples
        else:
            if range is None:
                self.original_images = [
                    os.path.join(original_dir, f)
                    for f in os.listdir(original_dir)
                    if f.lower().endswith((".jpg", ".png"))
                ]
                self.modified_images = [
                    os.path.join(modified_dir, f)
                    for f in os.listdir(modified_dir)
                    if f.lower().endswith((".jpg", ".png"))
                ]

            else:  # range is not None:
                indices = range  # Assuming range is a list of indices

                # Function to extract the numeric part from the file name
                def extract_number(fname):
                    import re

                    numbers = re.findall(r"\d+", fname)
                    return int(numbers[0]) if numbers else -1

                # Sort the files based on the numeric part of their names
                orig_map = {
                    extract_number(fname): fname for fname in os.listdir(original_dir)
                }
                chosen_nums = [n for n in indices if n in orig_map]

                # Select only the specified indices
                self.original_images = [
                    os.path.join(original_dir, orig_map[n]) for n in chosen_nums
                ]
                self.modified_images = [
                    os.path.join(modified_dir, orig_map[n])
                    for n in chosen_nums
                    if n in orig_map
                ]
            if mask_dir is not None:
                self.all_samples = [
                    (path, 0, None) for path in self.modified_images
                ] + [
                    (path, 1, os.path.join(mask_dir, os.path.basename(path)))
                    for path in self.original_images
                ]
            else:
                self.all_samples = [
                    (path, 0, None) for path in self.modified_images
                ] + [(path, 1, None) for path in self.original_images]

        self.imgs = self.all_samples  # for backward compatibility
        self.augment = augment
        self.return_filenames = return_filenames

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.all_samples[idx]

        # Load image and convert to RGB.
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
            angles = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
            angle = random.choice(angles)
            img = TF.rotate(img, angle, fill=[255, 255, 255])
            mask = TF.rotate(mask, angle, fill=0)

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

        if self.return_filenames:
            # Return the image, label, mask, and filename
            return image, torch.tensor(label, dtype=torch.long), mask_tensor, img_path

        return image, torch.tensor(label, dtype=torch.long), mask_tensor


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def load_data(
    original_path,
    modified_path,
    mask_path=None,
    batch_size=32,
    img_transform=None,
    mask_transform=None,
    num_workers=4,
    range=None,
    return_filenames=False,
):
    """
    Load the dataset and return a DataLoader.
    """
    dataset = CustomDataset(
        original_dir=original_path,
        modified_dir=modified_path,
        mask_dir=mask_path,
        image_transform=img_transform,
        mask_transform=mask_transform,
        augment=False,
        range=range,
        return_filenames=return_filenames,
    )
    return get_dataloader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def load_data_train_test(
    train_original_path,
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
    num_workers=4,
    range=None,
    return_filenames=False,
):
    test_loader = load_data(
        original_path=test_original_path,
        modified_path=test_modified_path,
        mask_path=mask_path_test,
        batch_size=batch_size,
        img_transform=image_transform_test,
        mask_transform=mask_transform_test,
        num_workers=num_workers,
        range=range,
        return_filenames=return_filenames,
    )
    train_loader = load_data(
        original_path=train_original_path,
        modified_path=train_modified_path,
        mask_path=mask_path_train,
        batch_size=batch_size,
        img_transform=image_transform_train,
        mask_transform=mask_transform_train,
        num_workers=num_workers,
        range=range,
        return_filenames=return_filenames,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_original_path = "data/train/configuration1"
    train_modified_path = "data/train/configuration2"
    test_original_path = "data/test/configuration1"
    test_modified_path = "data/test/configuration2"
    train_mask_path = "data/mask/train/mask2"
    test_mask_path = "data/mask/test/mask2"

    # Define transforms for images and masks.
    image_transform_train = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.NEAREST_EXACT
            ),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #  std=[0.229, 0.224, 0.225]),
        ]
    )
    image_transform_test = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # For the mask, we usually only need resizing and conversion to tensor.
    mask_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # no normalization, retains a single channel
        ]
    )

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
        range=None,
    )

    # Test one batch to check outputs.
    for images, labels, masks in train_loader:
        print("Images shape:", images.shape)  # Expected: [batch, 3, 224, 224]
        print("Labels shape:", labels.shape)
        print("Masks shape:", masks.shape)  # Expected: [batch, 1, 224, 224]
        # Optionally, save one image and mask.

        cv2.imwrite(
            "output/image.png", images[0].permute(1, 2, 0).numpy()[:, :, ::-1] * 255
        )
        cv2.imwrite("output/mask.png", masks[0].permute(1, 2, 0).numpy() * 255)
        break
