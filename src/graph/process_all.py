from .arcs import read_arcs
from .coordinates import read_coordinates
import os
import re
from tqdm import tqdm
from .plot_routes import plot_routes


def process_all_solutions(
    arcs_folder,
    coordinates_folder,
    output_folder,
    bounds=(-1, 11, -1, 11),
    valid_range=None,
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(
        os.listdir(arcs_folder), desc="Processing files", unit="file", leave=False
    ):
        match = re.match(r"Arcs_(\w+)_\d+\.txt", filename)
        # if not match:
        # print(f"Skipped file (no match): {filename}")
        if match:
            number = int(match.group(1))
            if valid_range is None or number in valid_range:
                instance = match.group(1)
                arcs_file = os.path.join(arcs_folder, filename)
                coordinates_file = os.path.join(
                    coordinates_folder, f"Coordinates_{instance}.txt"
                )
                output_file = os.path.join(output_folder, f"Plot_{instance}.png")

                if os.path.exists(coordinates_file):
                    arcs = read_arcs(arcs_file)
                    coordinates, depot = read_coordinates(coordinates_file)
                    plot_routes(arcs, coordinates, depot, output_file, bounds)
                else:
                    print(f"Warning: Coordinates file {coordinates_file} not found.")
