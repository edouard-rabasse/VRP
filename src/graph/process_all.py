from .read_arcs import read_arcs
from .read_coordinates import read_coordinates
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
    type="original",
    show_labels=False,
):
    """Process all solutions by reading arcs and coordinates, and plotting the routes.

    Parameters:
        - arcs_folder (str): Path to the folder containing arcs files.
        - coordinates_folder (str): Path to the folder containing coordinates files.
        - output_folder (str): Path to the folder where output plots will be saved.
        - bounds (tuple, optional): Geographic bounds for the plot. Defaults to (-1, 11, -1, 11).
        - valid_range (list, optional): List of valid instance numbers to process. Defaults to None.
        - type (str, optional): Type of the solution (e.g., "original", "refined"). Defaults to "original".
        - show_labels (bool, optional): Whether to show labels on the plot. Defaults to False.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(arcs_folder):
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
                    arcs = read_arcs(arcs_file, type=type)
                    coordinates, depot = read_coordinates(coordinates_file, type=type)
                    plot_routes(
                        arcs,
                        coordinates,
                        depot,
                        output_file,
                        bounds,
                        route_type=type,
                        show_labels=show_labels,
                    )
                else:
                    print(f"Warning: Coordinates file {coordinates_file} not found.")
