"""
Processing module for image and VRP text data.
"""

from .base_processor import BaseProcessor, BaseProcessingResults, ProcessingError
from .image_processor import (
    ImageProcessor,
    ImageProcessingResults,
    ImageProcessingError,
)
from .txt_processor import TxtProcessor, TxtProcessingResults, TxtProcessingError
from .batch_processor import BatchProcessor

import torch
from omegaconf import DictConfig


# Legacy functions for backward compatibility
def process_image(
    cfg: DictConfig, model: torch.nn.Module, fname: str, device: torch.device
) -> float:
    """Legacy function for processing a single image."""
    processor = ImageProcessor(cfg, model, device)
    try:
        result = processor.process(fname)
        print(f"Loss: {result.loss:.4f}")
        return result.loss
    except ImageProcessingError as e:
        print(f"Error processing {fname}: {e}")
        return float("nan")


def process_txt_files(
    cfg: DictConfig,
    model: torch.nn.Module,
    coords_file: str,
    arcs_file: str,
    device: torch.device,
):
    """Legacy function for processing VRP text files."""
    processor = TxtProcessor(cfg, model, device)
    try:
        result = processor.process(coords_file, arcs_file)
        print(f"Generated heatmap with shape: {result.heatmap_shape}")
        return result
    except TxtProcessingError as e:
        print(f"Error processing files: {e}")
        return None


__all__ = [
    "BaseProcessor",
    "BaseProcessingResults",
    "ProcessingError",
    "ImageProcessor",
    "ImageProcessingResults",
    "ImageProcessingError",
    "TxtProcessor",
    "TxtProcessingResults",
    "TxtProcessingError",
    "BatchProcessor",
    "process_image",
    "process_txt_files",
]
