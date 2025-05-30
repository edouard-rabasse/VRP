# graph_creator_modified.py: Read enhanced arc/coordinate data and plot routes with arc types and depot markings TODO: adapt to new structure
import sys
from pathlib import Path

# Add the root directory to the system path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import hydra
from omegaconf import DictConfig
from src.graph.read_arcs import read_arcs
from src.graph.read_coordinates import read_coordinates
from src.graph.plot_routes import plot_routes
from src.graph.process_all import process_all_solutions


@hydra.main(config_path="../../config/plot", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    numbers = cfg.numbers
    valid_range = range(cfg.valid_range[0], cfg.valid_range[1] + 1)
    bounds = tuple(cfg.bounds)
    for number in numbers:
        print("Processing configuration", number)
        arcs_folder = f"data/results_modified/configuration{number}/"
        coordinates_folder = "data/instances_modified/"
        output_folder = f"data/plots_modified/configuration{number}/"

        # Process all solutions
        process_all_solutions(
            arcs_folder,
            coordinates_folder,
            output_folder,
            bounds=bounds,
            valid_range=valid_range,
            type="modified",
            show_labels=True,
        )


if __name__ == "__main__":

    main()
