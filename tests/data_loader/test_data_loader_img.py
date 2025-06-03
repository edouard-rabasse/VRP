import os
import tempfile
import torch
import pytest
from PIL import Image
from torchvision.transforms import ToTensor

from src.data_loader_mask import CustomDataset, load_data, load_data_train_test


def create_dummy_image(path, size=(32, 32), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    img.save(path)


def create_dummy_mask(path, size=(32, 32), color=0):
    img = Image.new("L", size, color)
    img.save(path)


@pytest.fixture
def setup_image_dirs():
    with tempfile.TemporaryDirectory() as original_dir, tempfile.TemporaryDirectory() as modified_dir, tempfile.TemporaryDirectory() as mask_dir:
        # Create dummy original images and masks with filenames as indices for range testing
        for i in range(5):
            original_img_path = os.path.join(original_dir, f"{i}.png")
            create_dummy_image(original_img_path)
            original_mask_path = os.path.join(mask_dir, f"{i}.png")
            create_dummy_mask(original_mask_path)

        # Create dummy modified images similarly
        for i in range(5):
            modified_img_path = os.path.join(modified_dir, f"{i}.png")
            create_dummy_image(modified_img_path, color=(0, 255, 0))

        yield original_dir, modified_dir, mask_dir


def test_custom_dataset_len_and_getitem(setup_image_dirs):
    original_dir, modified_dir, mask_dir = setup_image_dirs
    dataset = CustomDataset(
        original_dir=original_dir,
        modified_dir=modified_dir,
        mask_dir=mask_dir,
        image_transform=ToTensor(),
        mask_transform=ToTensor(),
        augment=False,
        return_filenames=True,
    )

    assert len(dataset) == 10  # 5 original + 5 modified

    item = dataset[0]
    image, label, mask, filename = item
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(filename, str)
    assert label in (0, 1)


def test_custom_dataset_with_range_filters_files(setup_image_dirs):
    original_dir, modified_dir, mask_dir = setup_image_dirs
    requested_range = [1, 3]

    dataset = CustomDataset(
        original_dir=original_dir,
        modified_dir=modified_dir,
        mask_dir=mask_dir,
        range=requested_range,
    )

    sample_filenames = [os.path.basename(sample[0]) for sample in dataset.all_samples]
    sample_indices = sorted([int(os.path.splitext(f)[0]) for f in sample_filenames])

    # Assert only files with indices 1 and 3 are loaded
    assert all(i in requested_range for i in sample_indices)
    assert set(sample_indices) == set(requested_range)


def test_load_data_returns_dataloader_with_range(setup_image_dirs):
    original_dir, modified_dir, mask_dir = setup_image_dirs
    requested_range = [0, 2, 4]

    loader = load_data(
        original_path=original_dir,
        modified_path=modified_dir,
        mask_path=mask_dir,
        batch_size=1,
        range=requested_range,
        img_transform=ToTensor(),
        mask_transform=ToTensor(),
        num_workers=0,
        augment=False,
    )

    filenames = [os.path.basename(s[0]) for s in loader.dataset.all_samples]
    indices = sorted([int(os.path.splitext(f)[0]) for f in filenames])

    assert set(indices).issubset(set(requested_range))


def test_load_data_train_test_returns_two_loaders_with_range(setup_image_dirs):
    original_dir, modified_dir, mask_dir = setup_image_dirs
    requested_range = [0, 4]

    train_loader, test_loader = load_data_train_test(
        train_original_path=original_dir,
        test_original_path=original_dir,
        train_modified_path=modified_dir,
        test_modified_path=modified_dir,
        mask_path_train=mask_dir,
        mask_path_test=mask_dir,
        batch_size=1,
        image_transform_train=ToTensor(),
        image_transform_test=ToTensor(),
        mask_transform_train=ToTensor(),
        mask_transform_test=ToTensor(),
        num_workers=0,
        range=requested_range,
        augment=False,
    )

    def get_indices(loader):
        return sorted(
            [
                int(os.path.splitext(os.path.basename(s[0]))[0])
                for s in loader.dataset.all_samples
            ]
        )

    train_indices = get_indices(train_loader)
    test_indices = get_indices(test_loader)

    assert set(train_indices).issubset(set(requested_range))
    assert set(test_indices).issubset(set(requested_range))
