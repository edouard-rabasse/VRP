# evaluate_seg.py : evaluate if the heatmap is correct
import torch


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.visualization import get_heatmap, show_mask_on_image


def compute_bce_with_logits_mask(
    heatmap_logits: torch.Tensor, mask: torch.Tensor
) -> float:
    """
    Resize the binary `mask` to the spatial size of `heatmap_logits` and compute
    a pixel-wise BCEWithLogitsLoss.

    Args:
      heatmap_logits (torch.Tensor): raw logits for the positive class,
        shape can be [H, W], [1, H, W], or [1, 1, H, W].
      mask (torch.Tensor): binary mask, shape [H0, W0] or [1, H0, W0].

    Returns:
      float: the average BCEWithLogitsLoss over all pixels.
    """
    # 1) Normalize dimensions: ensure mask is [1,1,H0,W0]
    if mask.dim() == 2:
        m = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        # either [1,H0,W0] or [C,H,W]
        if mask.shape[0] == 1:
            m = mask.unsqueeze(0)
        else:
            m = mask[0:1].unsqueeze(0)
    else:
        # assume [1,1,H0,W0]
        m = mask

    # 2) Normalize heatmap: ensure [1,1,H,W]
    hm = heatmap_logits
    if isinstance(hm, np.ndarray):
        hm = torch.from_numpy(hm)
    if hm.dim() == 2:
        hm = hm.unsqueeze(0).unsqueeze(0)
    elif hm.dim() == 3:
        # [1,H,W]
        hm = hm.unsqueeze(1)
    # else assume [1,1,H,W]

    # 3) Resize mask to heatmap size using nearest/neighbour or max pooling
    H, W = hm.shape[-2:]
    m_resized = F.interpolate(m.float(), size=(H, W), mode="nearest")

    # 4) Compute BCEWithLogitsLoss
    loss = nn.BCEWithLogitsLoss(reduction="mean")(hm, m_resized)
    return loss.item()


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
