import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.transform import ProportionalThresholdResize


def compute_bce_with_logits_mask(
    heatmap_logits: torch.Tensor,
    mask: torch.Tensor,
    criterion: nn.Module = nn.BCEWithLogitsLoss(reduction="mean"),
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
    m_resized = ProportionalThresholdResize(size=(H, W))(m)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(m_resized.squeeze().cpu().numpy(), cmap="gray")
    # plt.title("Resized Mask")
    # plt.subplot(1, 2, 2)
    # plt.imshow(hm.squeeze().cpu().numpy(), cmap="hot")
    # plt.title("Heatmap Logits")
    # plt.show()

    # 4) Compute BCEWithLogitsLoss
    loss = criterion(hm, m_resized)
    return loss.item()


if __name__ == "__main__":
    # Example usage
    heatmap_logits = torch.randn(1, 1, 64, 64)  # Example heatmap logits
    mask = torch.randint(0, 2, (32, 32))  # Example binary mask

    loss = compute_bce_with_logits_mask(heatmap_logits, mask)
    print(f"BCE with logits loss: {loss:.4f}")
