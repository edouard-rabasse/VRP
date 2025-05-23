# evaluate_seg.py : evaluate if the heatmap is correct


import os, sys
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

from src.models import load_model
from src.transform import image_transform_test, mask_transform, denormalize
from src.visualization import get_heatmap, 

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    