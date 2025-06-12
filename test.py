import os
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models import load_model
from src.transform import image_transform_test, mask_transform

from src.utils.config_utils import load_selection_config
from src.data_loader_mask import load_data
from evaluate_seg import compute_seg_loss_from_loader


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(cfg.solver.java_lib)


if __name__ == "__main__":
    main()
