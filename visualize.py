"""
visualize.py
This script visualizes heatmaps and overlays for a given dataset using a pre-trained model.
It loads a model specified in the configuration, processes each image in the test dataset,
computes the loss, and saves the resulting heatmaps to the specified output directory.

Usage:
    Run this script as a standalone module. Configuration is handled via Hydra.

    For image processing:
        python visualize.py

    For VRP text file processing:
        python visualize.py processor_type=txt coords_dir=path/to/coords arcs_dir=path/to/arcs
"""

import hydra
from omegaconf import DictConfig

from src.visualization.runner import VisualizationRunner


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Main function for visualization processing.

    Args:
        cfg: Hydra configuration object
    """
    print("[Viz] Starting visualization pipeline...")

    try:
        # Create runner
        runner = VisualizationRunner(cfg)

        # Determine processing type
        processor_type = getattr(cfg, "processor_type", "image")

        if processor_type == "image":
            runner.run_image_processing()
        elif processor_type == "txt":
            runner.run_txt_processing()
        else:
            print(f"[Error] Unknown processor type: {processor_type}")
            print("Valid options: 'image', 'txt'")
            return

        print("[Viz] Pipeline completed successfully!")

    except Exception as e:
        print(f"[Error] Visualization pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
