import os
import tempfile
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import pytest
from src.data_loader_graph import (
    VRPGraphDataset,
    get_graph_dataloader,
    split_graph_dataset,
)


def write_file(path, lines):
    with open(path, "w") as f:
        f.writelines(line + "\n" for line in lines)


def create_dummy_arcs_file(path):
    lines = [
        "0;1;1;0",
        "1;2;2;0",
        "2;0;1;0",
    ]
    write_file(path, lines)


def create_dummy_coords_file(path):
    lines = [
        "0,0.0,0.0,0.0",
        "1,1.0,0.0,0.0",
        "2,0.5,1.0,0.0",
    ]
    write_file(path, lines)


@pytest.fixture
def setup_vrp_graph_dataset(num=3):
    with tempfile.TemporaryDirectory() as orig_dir, tempfile.TemporaryDirectory() as mod_dir, tempfile.TemporaryDirectory() as coords_dir:
        for i in range(num):  # Crée 3 instances
            orig_arcs = os.path.join(orig_dir, f"Arcs_{i}_0.txt")
            mod_arcs = os.path.join(mod_dir, f"Arcs_{i}_0.txt")
            coords = os.path.join(coords_dir, f"Coordinates_{i}.txt")

            create_dummy_arcs_file(orig_arcs)
            create_dummy_arcs_file(mod_arcs)
            create_dummy_coords_file(coords)

        yield orig_dir, mod_dir, coords_dir


def test_len_and_getitem(setup_vrp_graph_dataset):
    orig_dir, mod_dir, coords_dir = setup_vrp_graph_dataset

    ds = VRPGraphDataset(
        orig_dir,
        mod_dir,
        coords_dir,
        image_transform=ToTensor(),
        mask_transform=ToTensor(),
    )
    # On doit avoir 2 exemples (original + modifié)
    assert len(ds) == 2 * len(ds.instances)

    for i in range(len(ds)):
        img_t, label_t, mask_t = ds[i]
        assert isinstance(img_t, torch.Tensor)
        assert isinstance(label_t, torch.Tensor)
        assert isinstance(mask_t, torch.Tensor)
        assert label_t.item() in (0, 1)
        # Image tensor doit être 3xHxW
        assert img_t.ndim == 3 and img_t.shape[0] == 3
        # Mask tensor doit être 1xHxW
        assert mask_t.ndim == 3 and mask_t.shape[0] == 1


def test_get_graph_dataloader_and_split(setup_vrp_graph_dataset):
    orig_dir, mod_dir, coords_dir = setup_vrp_graph_dataset

    ds = VRPGraphDataset(orig_dir, mod_dir, coords_dir)

    loader = get_graph_dataloader(
        orig_dir, mod_dir, coords_dir, batch_size=2, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    images, labels, masks = batch
    assert images.shape[0] == 2
    assert labels.shape[0] == 2
    assert masks.shape[0] == 2

    train_ds, test_ds = split_graph_dataset(ds, train_ratio=0.5, seed=123)
    assert len(train_ds) + len(test_ds) == len(ds)
    assert set(i[0] for i in train_ds.instances).isdisjoint(
        set(i[0] for i in test_ds.instances)
    )


def test_split_graph_dataset_keeps_instance_pairs_together(setup_vrp_graph_dataset):
    orig_dir, mod_dir, coords_dir = setup_vrp_graph_dataset

    dataset = VRPGraphDataset(orig_dir, mod_dir, coords_dir)
    train_ds, test_ds = split_graph_dataset(dataset, train_ratio=0.5, seed=123)

    # Extraire les numéros d’instances pour chaque split
    train_instance_nums = set(inst[0] for inst in train_ds.instances)
    test_instance_nums = set(inst[0] for inst in test_ds.instances)

    # Les ensembles doivent être disjoints
    assert train_instance_nums.isdisjoint(test_instance_nums)

    # Pour chaque split, vérifier que les deux exemples (original et modifié) sont présents par instance
    # Ici, len(dataset) = 2 * nombre d’instances, donc on vérifie que les instances ne sont pas cassées
    for ds in [train_ds, test_ds]:
        for inst_num in set(inst[0] for inst in ds.instances):
            # Compter occurrences de cette instance (doit être 1 car instance stockée une fois, chaque instance donne 2 exemples)
            count = sum(1 for inst in ds.instances if inst[0] == inst_num)
            assert count == 1, f"L'instance {inst_num} est dupliquée dans le split"
