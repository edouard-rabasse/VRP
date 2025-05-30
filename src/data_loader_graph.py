import os, re, io
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.graph import read_arcs, read_coordinates
from src.mask import get_mask, get_mask_pixelised, get_removed_lines


class VRPGraphDataset(Dataset):
    def __init__(
        self,
        orig_arcs_folder: str,
        mod_arcs_folder: str,
        coords_folder: str,
        bounds: tuple = (-1, 11, -1, 11),
        pixel_size: int = 1,
        mask_method: str = "removed_lines",
        image_transform=None,
        mask_transform=None,
        valid_range=None,
    ):
        self.orig_folder = orig_arcs_folder
        self.mod_folder = mod_arcs_folder
        self.coords_folder = coords_folder
        self.bounds = bounds
        self.pixel_size = pixel_size
        self.mask_method = mask_method
        self.image_transform = image_transform or transforms.ToTensor()
        self.mask_transform = mask_transform or transforms.ToTensor()

        # match instance files
        pat = re.compile(r"Arcs_(\w+)_\d+\.txt")
        instances = []
        for fn in os.listdir(self.orig_folder):
            m = pat.match(fn)
            if not m:
                continue
            inst = m.group(1)
            orig_fp = os.path.join(self.orig_folder, fn)
            mod_fp = os.path.join(self.mod_folder, fn)
            coords_fn = f"Coordinates_{inst}.txt"
            coords_fp = os.path.join(self.coords_folder, coords_fn)
            if os.path.exists(mod_fp) and os.path.exists(coords_fp):
                instances.append((inst, orig_fp, mod_fp, coords_fp))

        if valid_range is not None:
            instances = [t for t in instances if int(t[0]) in valid_range]

        self.instances = instances

    def __len__(self):
        return len(self.instances) * 2

    def _plot_arcs(self, arcs, coords, depot, ax):
        ax.set_aspect("equal")
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        for t, h, mode, _ in arcs:
            x1, y1 = coords[t]
            x2, y2 = coords[h]
            color = (0, 1, 0) if mode == 2 else (0, 0, 1)
            ax.plot([x1, x2], [y1, y2], linestyle="-", color=color, linewidth=4)
        for node, (x, y) in coords.items():
            marker = "s" if node == depot else "o"
            ax.scatter(x, y, color=(1, 0, 0), marker=marker, s=60)
        ax.axis("off")

    def _render_image(self, arcs, coords, depot):
        fig, ax = plt.subplots(figsize=(10, 10))
        self._plot_arcs(arcs, coords, depot, ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        buf.close()
        return image

    def __getitem__(self, idx):
        inst, orig_fp, mod_fp, coords_fp = self.instances[idx // 2]
        label = idx % 2  # 1 = original, 0 = modified
        arcs_fp = orig_fp if label == 1 else mod_fp

        arcs = read_arcs(arcs_fp)
        coords, depot = read_coordinates(coords_fp)
        pil_img = self._render_image(arcs, coords, depot)

        if label == 0:
            mask = Image.new("L", pil_img.size, 0)
        else:
            orig_img = self._render_image(read_arcs(orig_fp), coords, depot)
            orig_arr = np.array(orig_img)
            mod_arr = np.array(pil_img)
            if self.mask_method == "default":
                mask_np = get_mask(orig_arr, mod_arr)
            elif self.mask_method == "removed_lines":
                mask_np = get_removed_lines(orig_arr, mod_arr, colored=False)
            else:
                mask_np = get_mask_pixelised(
                    orig_arr, mod_arr, pixel_size=self.pixel_size
                )
            mask_np = mask_np[..., 0] if mask_np.ndim == 3 else mask_np
            mask = Image.fromarray(mask_np.astype(np.uint8), mode="L")

        img_t = self.image_transform(pil_img)
        mask_t = self.mask_transform(mask)
        return img_t, torch.tensor(label, dtype=torch.long), mask_t


def get_graph_dataloader(
    orig_arcs_folder,
    mod_arcs_folder,
    coords_folder,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    **dataset_kwargs,
):
    ds = VRPGraphDataset(
        orig_arcs_folder, mod_arcs_folder, coords_folder, **dataset_kwargs
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def split_graph_dataset(dataset: VRPGraphDataset, train_ratio=0.8, seed=42):
    """
    Split a VRPGraphDataset into train/test datasets, keeping (original, modified) pairs together.
    """
    import random

    # Step 1: Get list of instances
    instances = dataset.instances.copy()
    random.Random(seed).shuffle(instances)

    # Step 2: Split instances
    n_train = int(len(instances) * train_ratio)
    train_instances = instances[:n_train]
    test_instances = instances[n_train:]

    # Step 3: Build new datasets
    train_ds = VRPGraphDataset(
        orig_arcs_folder=dataset.orig_folder,
        mod_arcs_folder=dataset.mod_folder,
        coords_folder=dataset.coords_folder,
        bounds=dataset.bounds,
        pixel_size=dataset.pixel_size,
        mask_method=dataset.mask_method,
        image_transform=dataset.image_transform,
        mask_transform=dataset.mask_transform,
    )
    test_ds = VRPGraphDataset(
        orig_arcs_folder=dataset.orig_folder,
        mod_arcs_folder=dataset.mod_folder,
        coords_folder=dataset.coords_folder,
        bounds=dataset.bounds,
        pixel_size=dataset.pixel_size,
        mask_method=dataset.mask_method,
        image_transform=dataset.image_transform,
        mask_transform=dataset.mask_transform,
    )

    # Manually set instances
    train_ds.instances = train_instances
    test_ds.instances = test_instances

    return train_ds, test_ds
