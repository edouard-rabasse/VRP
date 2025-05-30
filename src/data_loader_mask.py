import os
import re
import random
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


def _get_filenames_by_index(dir_path: str, indices: List[int]) -> List[str]:
    def extract_number(fname):
        numbers = re.findall(r"\d+", fname)
        return int(numbers[0]) if numbers else -1

    files = [f for f in os.listdir(dir_path) if f.lower().endswith((".jpg", ".png"))]
    file_map = {extract_number(f): f for f in files}
    return [os.path.join(dir_path, file_map[i]) for i in indices if i in file_map]


def default_image_transform():
    return transforms.ToTensor()


def default_mask_transform():
    return transforms.ToTensor()


class CustomDataset(Dataset):
    def __init__(
        self,
        original_dir: Optional[str] = None,
        modified_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        image_transform=None,
        mask_transform=None,
        samples: Optional[List[Tuple[str, int, Optional[str]]]] = None,
        augment: bool = False,
        range: Optional[List[int]] = None,
        return_filenames: bool = False,
    ):
        self.image_transform = image_transform or default_image_transform()
        self.mask_transform = mask_transform or default_mask_transform()
        self.augment = augment
        self.return_filenames = return_filenames

        if samples is not None:
            self.all_samples = samples
        else:
            if range is not None:
                self.original_images = _get_filenames_by_index(original_dir, range)
                self.modified_images = _get_filenames_by_index(modified_dir, range)
            else:
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

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.all_samples[idx]

        img = Image.open(img_path).convert("RGB")
        if label == 0 or mask_path is None:
            mask = Image.new("L", img.size, 0)
        else:
            mask = Image.open(mask_path).convert("L")

        if self.augment:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.choice(range(0, 46, 5))
            img = TF.rotate(img, angle, fill=[255, 255, 255])
            mask = TF.rotate(mask, angle, fill=0)

        image = self.image_transform(img)
        mask_tensor = self.mask_transform(mask)

        if self.return_filenames:
            return image, torch.tensor(label, dtype=torch.long), mask_tensor, img_path
        return image, torch.tensor(label, dtype=torch.long), mask_tensor


def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
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
    dataset = CustomDataset(
        original_dir=original_path,
        modified_dir=modified_path,
        mask_dir=mask_path,
        image_transform=img_transform,
        mask_transform=mask_transform,
        range=range,
        return_filenames=return_filenames,
    )
    return get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)


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
