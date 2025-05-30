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
    """
    Dataset that generates route plots and corresponding masks on the fly.
    For each instance, emits two samples:
      label=0: original configuration1 plot (zero mask)
      label=1: modified configurationN plot (diff mask vs config1)
    """

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
        valid_range=None,  # optional filter by instance IDs
    ):
        self.orig_folder = orig_arcs_folder
        self.mod_folder = mod_arcs_folder
        self.coords_folder = coords_folder
        self.bounds = bounds
        self.pixel_size = pixel_size
        self.mask_method = mask_method
        self.image_transform = image_transform or transforms.ToTensor()
        self.mask_transform = mask_transform or transforms.ToTensor()

        # collect matching instance files
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
        # apply selection filter if provided
        if valid_range is not None:
            # valid_range is an iterable of int instance IDs
            instances = [t for t in instances if int(t[0]) in valid_range]
        self.instances = instances

    def __len__(self):
        return len(self.instances) * 2

    def __getitem__(self, idx):
        inst, orig_fp, mod_fp, coords_fp = self.instances[idx // 2]
        label = idx % 2  # 1=original,0=modified
        arcs_fp = orig_fp if label == 1 else mod_fp

        # read arcs and coordinates

        arcs = read_arcs(arcs_fp)
        coords, depot = read_coordinates(coords_fp)

        # plot the arcs into a PIL image buffer
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        for t, h, mode, _ in arcs:
            x1, y1 = coords[t]
            x2, y2 = coords[h]
            linestyle = "-"
            color = (0, 1, 0) if mode == 2 else (0, 0, 1)
            ax.plot(
                [x1, x2],
                [y1, y2],
                linestyle=linestyle,
                color=color,
                linewidth=4,
                zorder=1,
            )
        red = (1, 0, 0)
        for node, (x, y) in coords.items():
            marker = "s" if node == depot else "o"
            ax.scatter(x, y, color=red, marker=marker, s=60, zorder=2)
        ax.axis("off")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        pil_img = Image.open(buf).convert("RGB")
        buf.close()

        # build mask
        if label == 0:
            mask = Image.new("L", pil_img.size, 0)
        else:
            # regenerate original image array
            buf2 = io.BytesIO()
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_xlim(self.bounds[0], self.bounds[1])
            ax2.set_ylim(self.bounds[2], self.bounds[3])
            orig_arcs = read_arcs(orig_fp)
            for t, h, mode, _ in orig_arcs:
                x1, y1 = coords[t]
                x2, y2 = coords[h]
                ax2.plot(
                    [x1, x2], [y1, y2], linestyle="-", color=(0, 0, 1), linewidth=4
                )
            ax2.scatter(coords[depot][0], coords[depot][1], color=red, marker="s", s=60)
            ax2.axis("off")
            fig2.savefig(buf2, format="png", bbox_inches="tight", pad_inches=0)
            plt.close(fig2)
            buf2.seek(0)
            orig_arr = np.array(Image.open(buf2).convert("RGB"))
            mod_arr = np.array(pil_img)
            buf2.close()

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

        # transforms to tensors
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
