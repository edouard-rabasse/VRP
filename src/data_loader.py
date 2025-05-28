"""
Unified data loader that chooses between mask-based and graph-based datasets.
"""

from src.data_loader_mask import load_data_train_test
from src.data_loader_graph import VRPGraphDataset
from torch.utils.data import random_split, DataLoader
import torch
from src.utils.config_utils import load_selection_config
from src.transform import image_transform_train, image_transform_test, mask_transform


def load_data(cfg):
    """
    Return (train_loader, test_loader) based on cfg.data.loader:
      - 'mask': uses load_data_train_test
      - 'graph': uses get_graph_dataloader for train and test
    """
    loader_type = cfg.data.loader
    if loader_type == "mask":
        # selection range for mask loader
        range_sel = load_selection_config(cfg.data)
        train_loader, test_loader = load_data_train_test(
            train_original_path=cfg.data.train_original_path,
            test_original_path=cfg.data.test_original_path,
            train_modified_path=cfg.data.train_modified_path,
            test_modified_path=cfg.data.test_modified_path,
            mask_path_train=cfg.data.train_mask_path,
            mask_path_test=cfg.data.test_mask_path,
            batch_size=cfg.batch_size,
            image_transform_train=image_transform_train(tuple(cfg.image_size)),
            image_transform_test=image_transform_test(tuple(cfg.image_size)),
            mask_transform_train=mask_transform(tuple(cfg.mask_shape)),
            mask_transform_test=mask_transform(tuple(cfg.mask_shape)),
            num_workers=cfg.num_workers if hasattr(cfg, "num_workers") else 4,
            range=range_sel,
        )
        return train_loader, test_loader
    elif loader_type == "graph":
        # graph-based loader: build full dataset then random split
        # apply optional instance selection
        sel_range = load_selection_config(cfg.data)
        print(
            cfg.data.orig_arcs_folder, cfg.data.mod_arcs_folder, cfg.data.coords_folder
        )
        full_ds = VRPGraphDataset(
            orig_arcs_folder=cfg.data.orig_arcs_folder,
            mod_arcs_folder=cfg.data.mod_arcs_folder,
            coords_folder=cfg.data.coords_folder,
            bounds=tuple(cfg.data.bounds),
            pixel_size=cfg.data.pixel_size,
            mask_method=cfg.data.mask_method,
            image_transform=image_transform_train(tuple(cfg.image_size)),
            mask_transform=mask_transform(tuple(cfg.mask_shape)),
            valid_range=sel_range,
        )
        total = len(full_ds)
        n_train = int(cfg.data.train_ratio * total)
        rng = torch.Generator().manual_seed(cfg.data.seed)
        train_ds, test_ds = random_split(
            full_ds, [n_train, total - n_train], generator=rng
        )
        # use single-process loading to avoid backend memory issues
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )
        return train_loader, test_loader
    else:
        raise ValueError(f"Unknown data.loader: {loader_type}")
