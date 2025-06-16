# process_image.py: Process a single image and its mask, compute heatmap, and save results
"""Processes a single image and its corresponding mask by computing a heatmap, overlaying the heatmap on the image, saving the results, and calculating the loss.
Args:
    cfg (omegaconf.DictConfig): Configuration object containing parameters for processing, including heatmap method, directories, and arguments.
    model (torch.nn.Module): The trained model used to compute the heatmap.
    fname (str): The filename or path of the image to process.
    device (torch.device or str): The device on which computations are performed (e.g., 'cpu' or 'cuda').
Returns:
    float: The computed loss value between the heatmap and the mask.
Workflow:
    1. Loads and transforms the image and its mask.
    2. Computes the heatmap using the specified method and model.
    3. Resizes the mask to match the heatmap dimensions.
    4. Overlays the heatmap on the image and saves the result.
    5. Reverses the heatmap and saves the arcs.
    6. Computes and prints the loss between the heatmap and the mask.
"""

import cv2
import torchvision.transforms.functional as TF


from src.models import load_model
from .get_heatmap import get_heatmap
from .show_mask_on_image import show_mask_on_image
from .load_transform_image_name import load_and_transform_image_mask
from .reverse_heatmap import reverse_heatmap
from .save_overlay import save_overlay
from src.transform import ProportionalThresholdResize

from ..heatmap import compute_bce_with_logits_mask, penalized_heatmap_loss
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
    mask = ProportionalThresholdResize(size=(heatmap.shape[0], heatmap.shape[1]))(mask)

    overlay = show_mask_on_image(
        mask, heatmap, alpha=0.5, interpolation=cv2.INTER_NEAREST
    )
    save_overlay(overlay, cfg.heatmap_dir, fname)

    reverse_heatmap(cfg, fname, heatmap)

    loss = compute_bce_with_logits_mask(heatmap, mask)
    print(loss)
    return loss
