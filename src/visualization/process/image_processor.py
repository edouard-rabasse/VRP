"""
Image processor for handling image and mask files.
"""

import time
import torch
from typing import Tuple
from dataclasses import dataclass
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from typing import Tuple, Optional

import cv2
from ..show_mask_on_image import show_mask_on_image
from ..save_overlay import save_overlay
from ...heatmap import compute_bce_with_logits_mask, penalized_heatmap_loss


from .base_processor import BaseProcessor, BaseProcessingResults, ProcessingError


class ImageProcessingError(ProcessingError):
    """Exception for image processing errors."""

    pass


@dataclass
class ImageProcessingResults(BaseProcessingResults):
    """Results from image processing."""

    filename: str
    loss: float
    mask_shape: Tuple[int, ...]
    overlay_saved: bool
    arcs_saved: bool

    def __init__(
        self,
        filename: str,
        loss: float,
        mask_shape: Tuple[int, ...],
        overlay_saved: bool,
        arcs_saved: bool,
        heatmap_shape: Tuple[int, ...],
        processing_time: float,
        heatmap_computed: bool,
        score: Optional[float] = None,
    ):
        # Initialize parent class with source = filename
        super().__init__(
            source=filename,
            heatmap_shape=heatmap_shape,
            processing_time=processing_time,
            heatmap_computed=heatmap_computed,
            score=score,
        )
        # Initialize own fields
        self.filename = filename
        self.loss = loss
        self.mask_shape = mask_shape
        self.overlay_saved = overlay_saved
        self.arcs_saved = arcs_saved


class ImageProcessor(BaseProcessor):
    """Handles processing of image and mask files."""

    def __init__(self, cfg: DictConfig, model: torch.nn.Module, device: torch.device):
        super().__init__(cfg, model, device)
        from ..load_transform_image_name import ImageMaskLoader

        self.image_loader = ImageMaskLoader(cfg, device)

    def process(self, filename: str) -> ImageProcessingResults:
        """
        Process a single image through the complete pipeline.

        Args:
            filename: Name of the image file to process

        Returns:
            ImageProcessingResults object with processing metrics

        Raises:
            ImageProcessingError: If processing fails
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Load and transform image and mask
            image_tensor, mask_tensor = self._load_or_generate_image(filename)

            non_normalized_image = self.image_loader.load_untransformed_image(
                self.cfg.data.test_original_path, filename
            )

            # Step 2: Compute heatmap
            heatmap = self._compute_heatmap(image_tensor)

            # step 2.5
            score = self._score(image_tensor)

            # Step 3: Prepare mask (resize if needed)
            # processed_mask = self._prepare_mask(mask_tensor, heatmap.shape)*
            processed_mask = mask_tensor

            name = filename.replace(".png", "").replace(".jpg", "")

            # Step 4: Create and save overlay
            overlay_saved = self._create_and_save_overlay(
                processed_mask, heatmap, f"{name}_overlay.png"
            )

            # Step 5: Save arcs (optional)
            arcs_saved = self._save_arcs(
                heatmap, filename.replace(".png", "").replace(".jpg", "")
            )

            # Step 6: Compute loss
            loss = self._compute_loss(heatmap, processed_mask)

            # Step 7: Save heatmap (optional)
            self._save_heatmap(heatmap, non_normalized_image, filename)

            processing_time = time.perf_counter() - start_time

            return ImageProcessingResults(
                filename=filename,
                loss=loss,
                heatmap_shape=heatmap.shape,
                mask_shape=processed_mask.shape,
                processing_time=processing_time,
                overlay_saved=overlay_saved,
                arcs_saved=arcs_saved,
                heatmap_computed=True,
                score=score,
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            raise ImageProcessingError(
                f"Failed to process {filename} after {processing_time:.2f}s: {str(e)}"
            ) from e

    def _load_or_generate_image(
        self, filename: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and transform image and mask."""
        try:
            return self.image_loader.load_image_and_mask(
                self.cfg.data.test_original_path, self.cfg.data.test_mask_path, filename
            )
        except Exception as e:
            raise ImageProcessingError(f"Failed to load image/mask: {e}") from e

    # def _prepare_mask(
    #     self, mask_tensor: torch.Tensor, heatmap_shape: Tuple[int, ...]
    # ) -> torch.Tensor:
    #     """Prepare mask by resizing if necessary."""
    #     try:
    #         if hasattr(self.cfg, "resize_mask") and self.cfg.resize_mask:
    #             from src.transform import ProportionalThresholdResize

    #             resizer = ProportionalThresholdResize(size=heatmap_shape[-2:])
    #             return resizer(mask_tensor)
    #         else:
    #             if mask_tensor.shape[-2:] != heatmap_shape[-2:]:
    #                 return TF.resize(
    #                     mask_tensor,
    #                     size=heatmap_shape[-2:],
    #                     interpolation=TF.InterpolationMode.NEAREST,
    #                 )
    #             return mask_tensor
    #     except Exception as e:
    #         raise ImageProcessingError(f"Failed to prepare mask: {e}") from e

    def _create_and_save_overlay(
        self, mask: torch.Tensor, heatmap: torch.Tensor, filename: str
    ) -> bool:
        """Create and save the overlay image."""
        try:

            alpha = getattr(self.cfg.heatmap, "overlay_alpha", 0.5)
            interpolation = getattr(
                self.cfg.heatmap, "interpolation", cv2.INTER_NEAREST
            )

            overlay = show_mask_on_image(
                mask, heatmap, alpha=alpha, interpolation=interpolation
            )
            save_overlay(overlay, str(self.output_dir), filename)
            return True

        except Exception as e:
            print(f"Warning: Failed to save mask overlay for {filename}: {e}")
            return False

    def _compute_loss(self, heatmap: torch.Tensor, mask: torch.Tensor) -> float:
        """Compute loss between heatmap and mask."""
        try:

            loss_method = getattr(self.cfg.heatmap, "loss_method", "bce")

            if loss_method == "bce":
                return compute_bce_with_logits_mask(heatmap, mask)
            elif loss_method == "penalized":
                return penalized_heatmap_loss(heatmap, mask)
            else:
                return compute_bce_with_logits_mask(heatmap, mask)

        except Exception as e:
            print(f"Warning: Failed to compute loss: {e}")
            return float("nan")
