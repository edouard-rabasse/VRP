# graph_creator.py: Read VRP arcs and coordinates, then plot and save route diagrams

import sys
from pathlib import Path

# Add the root directory to the system path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import hydra
from omegaconf import DictConfig
from src.graph.process_all import process_all_solutions


@hydra.main(config_path="../../config/plot", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    numbers = cfg.numbers
    valid_range = range(cfg.valid_range[0], cfg.valid_range[1] + 1)
    background_image = cfg.background_image if "background_image" in cfg else None

    print(cfg)
    bounds = tuple(cfg.bounds)
    for number in numbers:
        print("Processing configuration", number)
        arcs_folder = cfg.arcs_folder + f"configuration{number}/"
        coordinates_folder = cfg.coordinates_folder
        output_folder = cfg.output_folder + f"configuration{number}{cfg.special}/"

        # Process all solutions
        process_all_solutions(
            arcs_folder,
            coordinates_folder,
            output_folder,
            bounds=bounds,
            valid_range=valid_range,
            background_image=background_image,
        )


if __name__ == "__main__":
    # numbers = [7]
    # for number in numbers:
    #     print("Processing configuration", number)
    #     arcs_folder = f"data/results_modified/configuration{number}/"
    #     coordinates_folder = "data/instances_modified/"
    #     output_folder = f"data/plots_modified/configuration{number}/"
    #
    #     # Process all solutions
    #     process_all_solutions(arcs_folder, coordinates_folder, output_folder)
    main()
