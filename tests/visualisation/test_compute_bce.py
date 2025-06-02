import torch
import numpy as np
from src.visualization.compute_bce_with_logits_mask import compute_bce_with_logits_mask


def test_compute_bce_with_logits_mask_basic():
    # Create a simple heatmap with logits: all zeros (sigmoid(0) = 0.5)
    heatmap_logits = torch.zeros((1, 1, 4, 4))  # shape [1,1,4,4]

    # Create a mask of ones: expecting BCE loss for p=0.5 vs target=1
    mask = torch.ones((2, 2))  # shape [4,4]

    # BCE for p=0.5 and target=1: -log(0.5) = 0.693...
    expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        input=heatmap_logits, target=torch.ones_like(heatmap_logits)
    ).item()

    loss = compute_bce_with_logits_mask(heatmap_logits, mask)

    assert abs(loss - expected_loss) < 1e-5, f"Expected {expected_loss}, got {loss}"
