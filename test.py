# train.py : this script trains the model, all the parameters are in config.yaml and subfolders

import os, sys
import torch
from src.data_loader_mask import load_data_train_test

from src.transform import image_transform_train, image_transform_test, mask_transform
from src.utils.config_utils import load_selection_config
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}, CUDA={torch.cuda.is_available()}")

    # ── data loaders ────────────────────────────────────────────────────────
    print(cfg.data.selection.value)
    print("a")
    range = load_selection_config(cfg.data)
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
        # num_workers        = os.cpu_count(),
        num_workers=2,
        range=range,
    )
    print(
        f"[Train] Data loaded: {len(train_loader.dataset)} train / {len(test_loader.dataset)} test"
    )
    for i, (x, y, z) in enumerate(test_loader):
        print(f"batch {i} : {x.shape}, {y.shape}, {z.shape}")


if __name__ == "__main__":
    main()
