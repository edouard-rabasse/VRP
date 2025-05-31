# evaluate_seg.py : evaluate if the heatmap is correct
import torch


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.visualization import (
    get_heatmap,
    show_mask_on_image,
    compute_bce_with_logits_mask,
)


def compute_seg_loss_from_loader(
    data_loader, model, device, heatmap_method, heatmap_args
):
    loss = 0
    for imgs, labels, masks in data_loader:
        # imgs: [B,3,H,W], masks: [B,1,H,W], filenames: list of strings
        imgs = imgs.to(device)
        # ── batched heatmap ────────────────────────────────────────────────────

        # heatmaps: tensor [B, H, W] or [B,1,H,W] depending on implementation

        # ── per-sample postprocessing ──────────────────────────────────────────
        for img_tensor, mask_tensor in zip(imgs, masks):
            hm = get_heatmap(
                heatmap_method,
                model,
                img_tensor.unsqueeze(0),
                heatmap_args,
                device=device,
            )

            mask = mask_tensor

            loss += compute_bce_with_logits_mask(hm, mask)
    total = len(data_loader.dataset)
    return loss / total if total > 0 else 0
