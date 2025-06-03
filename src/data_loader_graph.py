import os
import re
import io
from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib

matplotlib.use("Agg")  # Backend non interactif
import matplotlib.pyplot as plt

from src.graph import read_arcs, read_coordinates, plot_routes
from src.mask import get_mask, get_mask_pixelised, get_removed_lines


class VRPGraphDataset(Dataset):
    def __init__(
        self,
        orig_arcs_folder: str,
        mod_arcs_folder: str,
        coords_folder: str,
        bounds: Tuple[float, float, float, float] = (-1, 11, -1, 11),
        pixel_size: int = 1,
        mask_method: str = "removed_lines",
        image_transform=None,
        mask_transform=None,
        valid_range: Optional[List[int]] = None,
    ):
        self.orig_folder = orig_arcs_folder
        self.mod_folder = mod_arcs_folder
        self.coords_folder = coords_folder
        self.bounds = bounds
        self.pixel_size = pixel_size
        self.mask_method = mask_method
        self.image_transform = image_transform or transforms.ToTensor()
        self.mask_transform = mask_transform or transforms.ToTensor()

        self.instances = self._find_instances(valid_range)

    def _find_instances(
        self, valid_range: Optional[List[int]]
    ) -> List[Tuple[str, str, str, str]]:
        pattern = re.compile(r"Arcs_(\d+)_\d+\.txt")
        instances = []
        for fname in os.listdir(self.orig_folder):
            match = pattern.match(fname)
            if not match:
                continue
            inst_num = int(match.group(1))
            if valid_range and inst_num not in valid_range:
                continue

            orig_fp = os.path.join(self.orig_folder, fname)
            mod_fp = os.path.join(self.mod_folder, fname)
            coords_fp = os.path.join(self.coords_folder, f"Coordinates_{inst_num}.txt")

            if os.path.exists(mod_fp) and os.path.exists(coords_fp):
                instances.append((str(inst_num), orig_fp, mod_fp, coords_fp))
        return instances

    def __len__(self):
        # Chaque instance a 2 exemples : original (label=1) et modifié (label=0)
        return len(self.instances) * 2

    def _render_image(self, arcs, coords, depot):
        buf = io.BytesIO()
        plot_routes(
            arcs=arcs,
            coordinates=coords,
            depot=depot,
            output_file=buf,  # au lieu d'un chemin fichier, on passe un buffer
            bounds=self.bounds,
            route_type="original",  # ou selon besoin
            show_labels=False,
        )
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        buf.close()
        return img

    def __getitem__(self, idx):
        inst_num, orig_fp, mod_fp, coords_fp = self.instances[idx // 2]
        label = idx % 2  # 0 = modifié, 1 = original
        arcs_fp = orig_fp if label == 1 else mod_fp

        arcs = read_arcs(arcs_fp)
        coords, depot = read_coordinates(coords_fp)

        img = self._render_image(arcs, coords, depot)

        if label == 0:
            mask = Image.new("L", img.size, 0)  # masque noir pour modifié
        else:
            orig_img = self._render_image(read_arcs(orig_fp), coords, depot)
            orig_arr = np.array(orig_img)
            mod_arr = np.array(img)
            if self.mask_method == "default":
                mask_np = get_mask(orig_arr, mod_arr)
            elif self.mask_method == "removed_lines":
                mask_np = get_removed_lines(orig_arr, mod_arr, colored=False)
            else:
                mask_np = get_mask_pixelised(
                    orig_arr, mod_arr, pixel_size=self.pixel_size
                )

            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            mask = Image.fromarray(mask_np.astype(np.uint8), mode="L")

        img_tensor = self.image_transform(img)
        mask_tensor = self.mask_transform(mask)
        return img_tensor, torch.tensor(label, dtype=torch.long), mask_tensor


def get_graph_dataloader(
    orig_arcs_folder: str,
    mod_arcs_folder: str,
    coords_folder: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs,
) -> DataLoader:
    dataset = VRPGraphDataset(
        orig_arcs_folder, mod_arcs_folder, coords_folder, **dataset_kwargs
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def split_graph_dataset(
    dataset: VRPGraphDataset,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[VRPGraphDataset, VRPGraphDataset]:
    import random

    instances = dataset.instances.copy()
    random.Random(seed).shuffle(instances)

    n_train = int(len(instances) * train_ratio)
    train_instances = instances[:n_train]
    test_instances = instances[n_train:]

    train_ds = VRPGraphDataset(
        dataset.orig_folder,
        dataset.mod_folder,
        dataset.coords_folder,
        bounds=dataset.bounds,
        pixel_size=dataset.pixel_size,
        mask_method=dataset.mask_method,
        image_transform=dataset.image_transform,
        mask_transform=dataset.mask_transform,
    )
    test_ds = VRPGraphDataset(
        dataset.orig_folder,
        dataset.mod_folder,
        dataset.coords_folder,
        bounds=dataset.bounds,
        pixel_size=dataset.pixel_size,
        mask_method=dataset.mask_method,
        image_transform=dataset.image_transform,
        mask_transform=dataset.mask_transform,
    )

    train_ds.instances = train_instances
    test_ds.instances = test_instances

    return train_ds, test_ds
