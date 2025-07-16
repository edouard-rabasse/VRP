"""
Base classes for all processors.
"""

import torch
import numpy as np
from typing import Tuple
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from ..get_heatmap import get_heatmap
from ...transform import image_transform_test
import cv2
from ..show_mask_on_image import show_mask_on_image
from ..save_overlay import save_overlay


class ProcessingError(Exception):
    """Base exception for processing errors."""

    pass


@dataclass
class BaseProcessingResults:
    """Base results from processing."""

    source: str
    heatmap_shape: Tuple[int, ...]
    processing_time: float
    heatmap_computed: bool


class BaseProcessor(ABC):
    """Base class for all processors."""

    def __init__(self, cfg: DictConfig, model: torch.nn.Module, device: torch.device):
        """
        Initialize the base processor.

        Args:
            cfg: Configuration object
            model: Trained model for heatmap computation
            device: Device for computations
        """
        self.cfg = cfg
        self.model = model
        self.device = device

        # Setup image transform
        self.image_transform = image_transform_test(cfg.image_size)

        # Ensure output directory exists if saving is enabled
        if hasattr(cfg, "heatmap_dir"):
            self.output_dir = Path(cfg.heatmap_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def process(self, *args, **kwargs) -> BaseProcessingResults:
        """Process input data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _load_or_generate_image(self, *args, **kwargs) -> torch.Tensor:
        """Load or generate image tensor. Must be implemented by subclasses."""
        pass

    def _compute_heatmap(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Compute heatmap using the specified method."""
        try:
            return get_heatmap(
                self.cfg.heatmap.method,
                self.model,
                image_tensor,
                self.cfg.heatmap.args,
                device=self.device,
            )
        except Exception as e:
            raise ProcessingError(f"Failed to compute heatmap: {e}") from e

    def _save_heatmap(
        self, heatmap: np.ndarray, image: torch.Tensor, name: str
    ) -> None:
        """Save the heatmap over the original image."""
        try:

            alpha = getattr(self.cfg.heatmap, "overlay_alpha", 0.5)
            interpolation = getattr(self.cfg.heatmap, "interpolation", cv2.INTER_LINEAR)

            overlay = show_mask_on_image(
                image, heatmap, alpha=alpha, interpolation=interpolation
            )
            save_overlay(overlay, self.output_dir, f"{name}_overlay.png")
        except Exception as e:
            raise ProcessingError(f"Failed to save heatmap overlay: {e}") from e

    def _save_arcs(self, heatmap: np.ndarray, name: str) -> bool:
        """Save arcs if enabled in configuration."""
        try:
            if hasattr(self.cfg.heatmap, "save_arcs") and self.cfg.heatmap.save_arcs:
                from ..reverse_heatmap import reverse_heatmap

                reverse_heatmap(self.cfg, f"{name}.txt", heatmap)
                return True
            return False
        except Exception as e:
            print(f"Warning: Failed to save arcs for {name}: {e}")
            return False

    def _score(self, tensor: torch.Tensor) -> float:
        try:
            self.model.eval()
            with torch.no_grad():
                out = self.model(tensor)
                score = torch.sigmoid(out).squeeze().cpu()[1].item()
            return score
        except Exception as e:
            raise ProcessingError(f"Failed to compute score: {e}") from e
