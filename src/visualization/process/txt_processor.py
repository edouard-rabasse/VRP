"""
Text processor for handling VRP coordinate and arc files.
"""

import time
import torch
import numpy as np
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig
from typing import Tuple


from .base_processor import BaseProcessor, BaseProcessingResults, ProcessingError


class TxtProcessingError(ProcessingError):
    """Exception for text file processing errors."""

    pass


@dataclass
class TxtProcessingResults(BaseProcessingResults):
    """Results from text file processing."""

    coords_file: str
    arcs_file: str
    plot_generated: bool
    arcs_saved: bool

    def __init__(
        self,
        coords_file: str,
        arcs_file: str,
        plot_generated: bool,
        arcs_saved: bool,
        heatmap_shape: Tuple[int, ...],
        processing_time: float,
        heatmap_computed: bool,
        score: Optional[float] = None,
    ):
        # Initialize parent class with combined source
        super().__init__(
            source=f"{coords_file} + {arcs_file}",
            heatmap_shape=heatmap_shape,
            processing_time=processing_time,
            heatmap_computed=heatmap_computed,
        )
        # Initialize own fields
        self.coords_file = coords_file
        self.arcs_file = arcs_file
        self.plot_generated = plot_generated
        self.arcs_saved = arcs_saved
        self.score = score


class TxtProcessor(BaseProcessor):
    """Handles processing of VRP coordinate and arc files."""

    def __init__(self, cfg: DictConfig, model: torch.nn.Module, device: torch.device):
        super().__init__(cfg, model, device)

        # Plot generation parameters
        self.bounds = getattr(cfg, "plot_bounds", (-1, 11, -1, 11))
        self.dpi = getattr(cfg, "plot_dpi", 100)
        self.route_type = getattr(cfg, "route_type", "original")
        self.show_labels = getattr(cfg, "show_labels", False)

    def process(
        self, coords_file: str, arcs_file: str, instance_name: Optional[str] = None
    ) -> TxtProcessingResults:
        """
        Process coordinate and arc files to generate image and compute heatmap.

        Args:
            coords_file: Path to coordinates file
            arcs_file: Path to arcs file
            instance_name: Optional name for the instance (for saving)

        Returns:
            TxtProcessingResults object with processing metrics

        Raises:
            TxtProcessingError: If processing fails
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Validate file paths
            self._validate_files(coords_file, arcs_file)

            # Step 2: Generate and transform image
            image_tensor = self._load_or_generate_image(coords_file, arcs_file)

            # Step 2.5 : score

            score = self._score(image_tensor)

            # Step 3: Compute heatmap
            heatmap = self._compute_heatmap(image_tensor)

            # Step 4: Save results (optional)
            arcs_saved = False
            if instance_name:
                arcs_saved = self._save_arcs(heatmap, instance_name)
                self._save_heatmap(heatmap, image_tensor, instance_name)

            processing_time = time.perf_counter() - start_time

            return TxtProcessingResults(
                coords_file=coords_file,
                arcs_file=arcs_file,
                heatmap_shape=heatmap.shape,
                processing_time=processing_time,
                plot_generated=True,
                heatmap_computed=True,
                arcs_saved=arcs_saved,
                score=score,
            )

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            raise TxtProcessingError(
                f"Failed to process {coords_file}, {arcs_file} after {processing_time:.2f}s: {str(e)}"
            ) from e

    def _load_or_generate_image(self, coords_file: str, arcs_file: str) -> torch.Tensor:
        """Generate plot from coordinate and arc files and convert to tensor."""
        try:
            # Generate plot
            plot_array = self._generate_plot(coords_file, arcs_file)

            # Transform to tensor
            from PIL import Image

            pil_image = Image.fromarray(plot_array)
            image_tensor = self.image_transform(pil_image).unsqueeze(0).to(self.device)

            return image_tensor
        except Exception as e:
            raise TxtProcessingError(f"Failed to generate image: {e}") from e

    def _validate_files(self, coords_file: str, arcs_file: str) -> None:
        """Validate that input files exist."""
        coords_path = Path(coords_file)
        arcs_path = Path(arcs_file)

        if not coords_path.exists():
            raise TxtProcessingError(f"Coordinates file not found: {coords_path}")
        if not arcs_path.exists():
            raise TxtProcessingError(f"Arcs file not found: {arcs_path}")

    def _generate_plot(self, coords_file: str, arcs_file: str) -> np.ndarray:
        """Generate plot from coordinate and arc files."""
        try:
            from ...graph.generate_plot import generate_plot_from_files

            plot_array = generate_plot_from_files(
                arcs_file=arcs_file,
                coords_file=coords_file,
                bounds=self.bounds,
                dpi=self.dpi,
                route_type=self.route_type,
                show_labels=self.show_labels,
                background_image=None,
            )
            return plot_array
        except Exception as e:
            raise TxtProcessingError(f"Failed to generate plot: {e}") from e
