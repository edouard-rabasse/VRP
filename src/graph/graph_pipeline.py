from .mask import process_image_pairs
from .train_split import split_dataset
from .graph_creator import process_all_solutions


import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig


class GraphPlotting:
    def __init__(self, config: DictConfig):
        self.cfg = config

    def run_graph_generation(self):
        """Step 1: Generate graph visualizations."""
        print("=== STEP 1: Generating Graphs ===")

        numbers = self.cfg.numbers
        valid_range = range(self.cfg.valid_range[0], self.cfg.valid_range[1] + 1)
        background_image = getattr(self.cfg, "background_image", None)
        bounds = tuple(self.cfg.bounds)

        for number in numbers:
            print(f"Processing configuration {number}")
            arcs_folder = self.cfg.arcs_folder + f"configuration{number}/"
            coordinates_folder = self.cfg.coordinates_folder
            output_folder = (
                self.cfg.output_folder + f"configuration{number}{self.cfg.special}/"
            )

            process_all_solutions(
                arcs_folder=arcs_folder,
                coordinates_folder=coordinates_folder,
                output_folder=output_folder,
                valid_range=valid_range,
                bounds=bounds,
                background_image=background_image,
            )

        print("✓ Graph generation completed")

    def run_mask_generation(self):
        """Step 2: Generate masks from original/modified image pairs."""
        print("\n=== STEP 2: Generating Masks ===")

        # Process each configuration
        for number in self.cfg.numbers:
            print(f"Processing masks for configuration {number}")

            original_dir = (
                self.cfg.output_folder
                + f"configuration{number}{self.cfg.special}/original/"
            )
            modified_dir = (
                self.cfg.output_folder
                + f"configuration{number}{self.cfg.special}/modified/"
            )

            # Generate different mask types
            mask_types = [
                ("mask_removed_color", "default", True),
                ("mask_removed", "default", False),
                ("mask_classic", "classic", False),
            ]

            for mask_name, method, colored in mask_types:
                print(f"  Generating {mask_name} masks...")
                output_dir = self.cfg.mask_output_folder + f"{mask_name}/"

                process_image_pairs(
                    original_dir=original_dir,
                    modified_dir=modified_dir,
                    output_dir=output_dir,
                    pixel_size=getattr(self.cfg, "pixel_size", 10),
                    method=method,
                    colored=colored,
                )

        print("✓ Mask generation completed")

    def run_train_test_split(self):
        """Step 3: Split datasets into train/test sets."""
        print("\n=== STEP 3: Splitting Train/Test Sets ===")

        # Split mask datasets
        src_dirs = ["mask_removed_color", "mask_removed", "mask_classic"]

        for src_dir in src_dirs:
            print(f"Splitting {src_dir} dataset...")
            split_dataset(
                src_dir=self.cfg.mask_output_folder + f"{src_dir}",
                dst_dir=self.cfg.split_output_folder + f"{src_dir}/",
                train_ratio=getattr(self.cfg, "train_ratio", 0.8),
                random_state=getattr(self.cfg, "random_state", 42),
            )

        # Split original graph images
        print("Splitting original graph dataset...")
        split_dataset(
            src_dir=self.cfg.output_folder,
            dst_dir=self.cfg.split_output_folder,
            train_ratio=getattr(self.cfg, "train_ratio", 0.8),
            random_state=getattr(self.cfg, "random_state", 42),
        )

        print("✓ Train/test splitting completed")

    def run_full_pipeline(self):
        """Run the complete pipeline: graphs → masks → splits."""
        print("🚀 Starting VRP Data Processing Pipeline")
        print("=" * 50)

        try:
            self.run_graph_generation()
            self.run_mask_generation()
            self.run_train_test_split()

            print("\n" + "=" * 50)
            print("🎉 Pipeline completed successfully!")
            print(f"📊 Final datasets available in: {self.cfg.split_output_folder}")

        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise


@hydra.main(
    config_path="../config/plot", config_name="data_processing", version_base=None
)
def main(cfg: DictConfig) -> None:
    """Main entry point for the plotting"""

    pipeline = GraphPlotting(cfg)

    # Check which steps to run
    steps = getattr(cfg, "steps", ["graphs", "masks", "splits"])

    if "graphs" in steps:
        pipeline.run_graph_generation()

    if "masks" in steps:
        pipeline.run_mask_generation()

    if "splits" in steps:
        pipeline.run_train_test_split()

    print("\n✨ Selected pipeline steps completed!")
