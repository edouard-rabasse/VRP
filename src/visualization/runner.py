"""
Visualization runner for handling different types of visualization tasks.
"""

import os
import torch
from pathlib import Path
from typing import List, Optional
import time
from omegaconf import DictConfig

from src.models import load_model
from .process import ImageProcessor, TxtProcessor, BatchProcessor


class VisualizationRunner:
    """Main runner for visualization tasks."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        """Load and return the configured model."""
        self.cfg.load_model = True
        model = load_model(self.cfg.model.name, self.device, self.cfg.model).eval()
        print(f"[Viz] Model loaded: {self.cfg.model.name} on {self.device}")
        return model

    def run_image_processing(self) -> None:
        """Run visualization for image files."""
        print(f"[Viz] Starting image processing...")

        # Setup processor
        processor = ImageProcessor(self.cfg, self.model, self.device)

        # Check if single image is specified
        single_image = getattr(self.cfg, "single_image", None)
        if single_image:
            self._process_single_image(processor, single_image)
            return

        # Get image files
        image_files = self._get_image_files()
        if not image_files:
            print("[Viz] No image files found!")
            return

        print(f"[Viz] Found {len(image_files)} image files")

        # Process images
        if getattr(self.cfg, "use_batch_processing", False):
            self._process_images_batch(processor, image_files)
        else:
            self._process_images_sequential(processor, image_files)

    def run_txt_processing(self) -> None:
        """Run visualization for VRP text files."""
        print(f"[Viz] Starting VRP text file processing...")

        # Setup processor
        processor = TxtProcessor(self.cfg, self.model, self.device)

        single_coords = getattr(self.cfg, "single_coords_file", None)
        single_arcs = getattr(self.cfg, "single_arcs_file", None)

        if single_coords and single_arcs:
            instance_name = getattr(self.cfg, "single_instance_name", "single_instance")
            self._process_single_txt_pair(
                processor, single_coords, single_arcs, instance_name
            )
            return

        # Get file pairs
        file_pairs = self._get_txt_file_pairs()
        if not file_pairs:
            print("[Viz] No matching coordinate/arc file pairs found!")
            return

        print(f"[Viz] Found {len(file_pairs)} coordinate/arc file pairs")

        # Process files
        if getattr(self.cfg, "use_batch_processing", False):
            self._process_txt_files_batch(processor, file_pairs)
        else:
            self._process_txt_files_sequential(processor, file_pairs)

    def _get_image_files(self) -> List[str]:
        """Get list of image files to process."""
        test_path = Path(self.cfg.data.test_original_path)
        if not test_path.exists():
            print(f"[Error] Test directory not found: {test_path}")
            return []

        # Support multiple image formats
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        image_files = [
            f.name
            for f in test_path.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

        return sorted(image_files)

    def _process_single_image(
        self, processor: ImageProcessor, image_filename: str
    ) -> None:
        """Process a single image file."""
        print(f"[Viz] Processing single image: {image_filename}")

        start_time = time.perf_counter()

        try:
            result = processor.process(image_filename)
            processing_time = time.perf_counter() - start_time

            print(f"[SUCCESS] {image_filename}:")
            print(f"  Loss: {result.loss:.4f}")
            print(f"  Heatmap shape: {result.heatmap_shape}")
            print(f"  Mask shape: {result.mask_shape}")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Overlay saved: {result.overlay_saved}")
            print(f"  Arcs saved: {result.arcs_saved}")

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            print(f"[ERROR] Failed to process {image_filename}: {e}")
            print(f"  Processing time: {processing_time:.2f}s")

    def _process_single_txt_pair(
        self,
        processor: TxtProcessor,
        coords_file: str,
        arcs_file: str,
        instance_name: str,
    ) -> None:
        """Process a single coordinate/arcs file pair."""
        print(f"[Viz] Processing single VRP instance: {instance_name}")
        print(f"  Coordinates: {coords_file}")
        print(f"  Arcs: {arcs_file}")

        start_time = time.perf_counter()

        try:
            result = processor.process(coords_file, arcs_file, instance_name)
            processing_time = time.perf_counter() - start_time

            print(f"[SUCCESS] {instance_name}:")
            print(f"  Heatmap shape: {result.heatmap_shape}")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Plot generated: {result.plot_generated}")
            print(f"  Arcs saved: {result.arcs_saved}")
            print(f"  Score: {result.score:.4f}")

        except Exception as e:
            processing_time = time.perf_counter() - start_time
            print(f"[ERROR] Failed to process {instance_name}: {e}")
            print(f"  Processing time: {processing_time:.2f}s")

    def _get_txt_file_pairs(self) -> List[tuple]:
        """Get list of (coords_file, arcs_file, instance_name) tuples."""
        coords_dir = getattr(self.cfg, "coords_dir", None)
        arcs_dir = getattr(self.cfg, "arcs_dir", None)

        if not coords_dir or not arcs_dir:
            print(
                "[Error] coords_dir and arcs_dir must be specified for txt processing"
            )
            return []

        coords_path = Path(coords_dir)
        arcs_path = Path(arcs_dir)

        if not coords_path.exists():
            print(f"[Error] Coordinates directory not found: {coords_path}")
            return []
        if not arcs_path.exists():
            print(f"[Error] Arcs directory not found: {arcs_path}")
            return []

        # Find matching files
        file_pairs = []
        coords_pattern = getattr(self.cfg, "coords_pattern", "Coordinates_*.txt")
        arcs_pattern = getattr(self.cfg, "arcs_pattern", "Arcs_*_*.txt")

        coords_files = list(coords_path.glob(coords_pattern))
        arcs_files = list(arcs_path.glob(arcs_pattern))

        for coords_file in coords_files:
            # Extract instance ID from coordinates filename
            # Expected format: Coordinates_{instance_id}.txt
            coords_name = coords_file.stem
            if coords_name.startswith("Coordinates_"):
                instance_id = coords_name.replace("Coordinates_", "")

                # Find corresponding arcs files
                matching_arcs = [
                    f for f in arcs_files if f.stem.startswith(f"Arcs_{instance_id}_")
                ]

                for arcs_file in matching_arcs:
                    # Extract suffix from arcs filename
                    arcs_name = arcs_file.stem
                    parts = arcs_name.split("_")
                    if len(parts) >= 3:
                        suffix = parts[2]
                        instance_name = f"instance_{instance_id}_{suffix}"
                        file_pairs.append(
                            (str(coords_file), str(arcs_file), instance_name)
                        )

        return sorted(file_pairs)

    def _process_images_sequential(
        self, processor: ImageProcessor, image_files: List[str]
    ) -> None:
        """Process images one by one."""
        total_loss = 0.0
        total_time = 0.0
        successful = 0

        for i, filename in enumerate(image_files, 1):
            start_time = time.perf_counter()

            try:
                result = processor.process(filename)
                processing_time = time.perf_counter() - start_time

                total_loss += (
                    result.loss if not torch.isnan(torch.tensor(result.loss)) else 0
                )
                total_time += processing_time
                successful += 1

                print(
                    f"[{i}/{len(image_files)}] {filename}: "
                    f"Loss={result.loss:.4f}, Time={processing_time:.2f}s"
                )
                print("score:", result.score)

            except Exception as e:
                processing_time = time.perf_counter() - start_time
                print(
                    f"[{i}/{len(image_files)}] ✗ {filename}: {e} (Time={processing_time:.2f}s)"
                )

        # Print summary
        avg_loss = total_loss / successful if successful > 0 else 0
        avg_time = total_time / len(image_files)

        print(f"\n[Viz] Summary:")
        print(f"  Processed: {successful}/{len(image_files)} images")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")

    def _process_images_batch(
        self, processor: ImageProcessor, image_files: List[str]
    ) -> None:
        """Process images using batch processor."""
        batch_processor = BatchProcessor(processor)
        results = batch_processor.process_batch(image_files)

        # Calculate average loss for successful images
        successful_results = [r for r in results["results"] if hasattr(r, "loss")]
        avg_loss = (
            sum(r.loss for r in successful_results) / len(successful_results)
            if successful_results
            else 0
        )

        print(f"\n[Viz] Batch Summary:")
        print(
            f"  Processed: {results['successful']}/{results['total_processed']} images"
        )
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Average time: {results['average_time']:.2f}s")
        print(f"  Total time: {results['total_time']:.2f}s")

    def _process_txt_files_sequential(
        self, processor: TxtProcessor, file_pairs: List[tuple]
    ) -> None:
        """Process VRP files one by one."""
        total_time = 0.0
        successful = 0

        for i, (coords_file, arcs_file, instance_name) in enumerate(file_pairs, 1):
            start_time = time.perf_counter()

            try:
                result = processor.process(coords_file, arcs_file, instance_name)
                processing_time = time.perf_counter() - start_time

                total_time += processing_time
                successful += 1

                print(
                    f"[{i}/{len(file_pairs)}] {instance_name}: "
                    f"Heatmap={result.heatmap_shape}, Time={processing_time:.2f}s"
                )

            except Exception as e:
                processing_time = time.perf_counter() - start_time
                print(
                    f"[{i}/{len(file_pairs)}] ✗ {instance_name}: {e} (Time={processing_time:.2f}s)"
                )

        # Print summary
        avg_time = total_time / len(file_pairs)

        print(f"\n[Viz] Summary:")
        print(f"  Processed: {successful}/{len(file_pairs)} file pairs")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")

    def _process_txt_files_batch(
        self, processor: TxtProcessor, file_pairs: List[tuple]
    ) -> None:
        """Process VRP files using batch processor."""
        batch_processor = BatchProcessor(processor)
        results = batch_processor.process_batch(file_pairs)

        print(f"\n[Viz] Batch Summary:")
        print(
            f"  Processed: {results['successful']}/{results['total_processed']} file pairs"
        )
        print(f"  Average time: {results['average_time']:.2f}s")
        print(f"  Total time: {results['total_time']:.2f}s")
