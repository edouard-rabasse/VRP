"""
Batch processor for handling multiple processing operations.
"""

from typing import Dict, Any
from .base_processor import BaseProcessor, ProcessingError
from .image_processor import ImageProcessor
from .txt_processor import TxtProcessor


class BatchProcessor:
    """Handles batch processing for any processor type."""

    def __init__(self, processor: BaseProcessor):
        self.processor = processor
        self.results = []

    def process_batch(self, inputs: list) -> Dict[str, Any]:
        """
        Process a batch of inputs.

        Args:
            inputs: List of inputs (format depends on processor type)

        Returns:
            Dictionary with batch processing results
        """
        successful = 0
        failed = 0
        processing_times = []

        for i, input_data in enumerate(inputs):
            try:
                # Handle different input formats
                if isinstance(self.processor, ImageProcessor):
                    result = self.processor.process(input_data)
                    identifier = input_data
                elif isinstance(self.processor, TxtProcessor):
                    coords_file, arcs_file = input_data[:2]
                    instance_name = (
                        input_data[2] if len(input_data) > 2 else f"batch_instance_{i}"
                    )
                    result = self.processor.process(
                        coords_file, arcs_file, instance_name
                    )
                    identifier = f"{coords_file} + {arcs_file}"
                else:
                    continue

                self.results.append(result)
                successful += 1
                processing_times.append(result.processing_time)

                print(f"✓ {identifier}: Time={result.processing_time:.2f}s")

            except ProcessingError as e:
                failed += 1
                print(f"✗ {identifier}: {e}")

        return {
            "total_processed": len(inputs),
            "successful": successful,
            "failed": failed,
            "average_time": (
                sum(processing_times) / len(processing_times) if processing_times else 0
            ),
            "total_time": sum(processing_times),
            "results": self.results,
        }
