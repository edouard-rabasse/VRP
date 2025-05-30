# process_image.py: Process a single image and its mask, compute heatmap, and save results

import cv2
import torchvision.transforms.functional as TF


from src.models import load_model
from .get_heatmap import get_heatmap
from .show_mask_on_image import show_mask_on_image
from .load_transform_image_name import load_and_transform_image_mask
from .reverse_heatmap import reverse_heatmap
from .save_overlay import save_overlay

from evaluate_seg import compute_bce_with_logits_mask
import hydra
from omegaconf import DictConfig
import time


def process_image(cfg, model, fname, device):
    """
    This function processes a single image, finds its mask, computes the heatmap,
    overlays the heatmap on the image, saves the arcs, and saves the results.
    """
    t_img, mask = load_and_transform_image_mask(cfg, fname, device)
    heatmap = get_heatmap(
        cfg.heatmap.method, model, t_img, cfg.heatmap.args, device=device
    )

    overlay = show_mask_on_image(
        mask, heatmap, alpha=0.5, interpolation=cv2.INTER_NEAREST
    )
    save_overlay(overlay, cfg.heatmap_dir, fname)

    reverse_heatmap(cfg, fname, heatmap)

    loss = compute_bce_with_logits_mask(heatmap, mask)
    return loss
