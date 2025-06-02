from .read_arcs import read_arcs
from .read_coordinates import read_coordinates
from .plot_routes import plot_routes
import os
import re
from tqdm import tqdm


def get_instance_number(filename):
    match = re.match(r"Arcs_(\w+)_\d+\.txt", filename)
    return match.group(1) if match else None


def should_process_instance(instance_number, valid_range):
    return valid_range is None or int(instance_number) in valid_range


def process_single_solution(
    arcs_file, coordinates_file, output_file, bounds, route_type, show_labels
):
    arcs = read_arcs(arcs_file, type=route_type)
    coordinates, depot = read_coordinates(coordinates_file, type=route_type)
    plot_routes(arcs, coordinates, depot, output_file, bounds, route_type, show_labels)


def process_all_solutions(
    arcs_folder,
    coordinates_folder,
    output_folder,
    bounds=(-1, 11, -1, 11),
    valid_range=None,
    type="original",
    show_labels=False,
):
    """Batch process all solution files."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in tqdm(os.listdir(arcs_folder), desc="Processing solutions"):
        instance = get_instance_number(filename)
        if instance and should_process_instance(instance, valid_range):
            arcs_file = os.path.join(arcs_folder, filename)
            coordinates_file = os.path.join(
                coordinates_folder, f"Coordinates_{instance}.txt"
            )
            output_file = os.path.join(output_folder, f"Plot_{instance}.png")

            if os.path.exists(coordinates_file):
                process_single_solution(
                    arcs_file, coordinates_file, output_file, bounds, type, show_labels
                )
            else:
                print(f"⚠️ Coordinates file not found: {coordinates_file}")
