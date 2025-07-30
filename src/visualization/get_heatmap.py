"""
Heatmap generation utilities for VRP visualization.

This module provides functions to generate heatmaps using various methods
including GradCAM, Grad Rollout, and multi-task approaches.
"""

import numpy as np
import torch
from typing import Dict, Any

from .get_attr import recursive_getattr
from .heatmap import GradCAM
from src.models.vit_explain.grad_rollout import VITAttentionGradRollout


def get_heatmap(
    method: str,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    args: Dict[str, Any],
    device: str = "cpu",
    percentile_for_thresh: int = 95,
) -> np.ndarray:
    """
    Generate a heatmap using the specified method.

    Args:
        method: The method to use for generating the heatmap.
            Supported methods: 'gradcam', 'grad_rollout', 'multi_task',
            'grad_cam_vgg', 'seg'
        model: The PyTorch model to use for generating the heatmap.
        input_tensor: The input tensor to the model.
        args: Additional arguments for the method.
        device: Device to run computation on ('cpu' or 'cuda').
        percentile_for_thresh: Percentile threshold for heatmap (currently unused).

    Returns:
        Generated heatmap as a numpy array normalized to [0, 1].

    Raises:        ValueError: If an unknown method is specified.
    """
    input_tensor = input_tensor.to(device)

    if method == "gradcam":
        return _generate_gradcam_heatmap(model, input_tensor, args, device)
    elif method == "grad_rollout":
        return _generate_grad_rollout_heatmap(model, input_tensor, args, device)
    elif method == "multi_task":
        return _generate_multi_task_heatmap(model, input_tensor)
    elif method == "grad_cam_vgg":
        return _generate_vgg_gradcam_heatmap(model, input_tensor, args)
    elif method == "seg":
        return _generate_segmentation_heatmap(model, input_tensor)
    else:
        raise ValueError(f"Unknown method: {method}")


def _generate_gradcam_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    args: Dict[str, Any],
    device: str,
) -> np.ndarray:
    """Generate heatmap using GradCAM method."""
    model.eval()
    target_layer = recursive_getattr(model, args.target_layer)
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam(input_tensor, class_index=args.class_index)
    return heatmap / (np.max(heatmap) + 1e-8)


def _generate_grad_rollout_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    args: Dict[str, Any],
    device: str,
) -> np.ndarray:
    """Generate heatmap using Grad Rollout method."""
    grad_rollout = VITAttentionGradRollout(
        model, discard_ratio=args.discard_ratio, device=device
    )
    heatmap = grad_rollout(input_tensor, category_index=args.class_index)
    return heatmap / (np.max(heatmap) + 1e-8)


def _generate_multi_task_heatmap(
    model: torch.nn.Module, input_tensor: torch.Tensor
) -> np.ndarray:
    """Generate heatmap using multi-task approach."""
    with torch.no_grad():
        model.eval()
        cls_logits, seg_logits = model(input_tensor)
        heatmap = seg_logits[0, 0]  # First channel of interest
        return torch.nn.functional.sigmoid(heatmap).cpu().numpy()


def _generate_vgg_gradcam_heatmap(
    model: torch.nn.Module, input_tensor: torch.Tensor, args: Dict[str, Any]
) -> np.ndarray:
    """Generate heatmap using GradCAM for VGG models."""
    model.eval()
    target_layer = model.features[29]  # Last convolutional layer
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam(input_tensor, class_index=args["class_index"])
    return heatmap / (np.max(heatmap) + 1e-8)


def _generate_segmentation_heatmap(
    model: torch.nn.Module, input_tensor: torch.Tensor
) -> np.ndarray:
    """Generate heatmap using segmentation model."""
    with torch.no_grad():
        model.eval()
        seg_logits = model(input_tensor)
        heatmap = seg_logits[0, 0]  # First channel of interest
        return torch.sigmoid(heatmap).cpu().numpy()
