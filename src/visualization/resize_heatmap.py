import cv2
import numpy as np


def resize_heatmap(
    heatmap: np.ndarray, target_size: tuple, interpolation=cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize heatmap to target dimensions with normalization.

    Args:
        heatmap: Input heatmap array
        target_size: Target (width, height) tuple
        interpolation: OpenCV interpolation method

    Returns:
        Resized and normalized heatmap array
    """
    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=interpolation)
    heatmap_resized = (heatmap_resized - np.min(heatmap_resized)) / (
        np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-8
    )

    return heatmap_resized
