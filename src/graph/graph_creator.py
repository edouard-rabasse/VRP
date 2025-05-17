import matplotlib.pyplot as plt
import numpy as np
import os
import re
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# plt.style.use('default')
# plt.rcParams['lines.alpha'] = 1.0


def read_arcs(file_path):
    arcs = []
    with open(file_path, "r") as file:
        for line in file:
            tail, head, mode, route_id = map(int, line.strip().split(";"))
            arcs.append((tail, head, mode, route_id))
    return arcs


def read_coordinates(file_path):
    coordinates = {}
    last_node = None
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            node = int(parts[0])
            x, y = map(float, parts[1:3])
            coordinates[node] = (x, y)
            last_node = node  # The last node is the depot
    return coordinates, last_node


def plot_routes(arcs, coordinates, depot, output_file, bounds=(-1, 11, -1, 11)):
    # Create a figure and axes with a 10x10 inch size and equal aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    # Remove the borders (spines) from the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove axis ticks and labels for a clean appearance
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Optionally, if you ever need to use colors per route, you can uncomment the following lines.
    # route_colors = {}
    # unique_routes = set(route_id for _, _, _, route_id in arcs)
    # colors = plt.cm.tab10(np.linspace(0, 1, len(unique_routes)))
    # for route_id, color in zip(unique_routes, colors):
    #     route_colors[route_id] = color

    # Plot each arc without adding a legend label (to avoid duplicate legends)
    for tail, head, mode, route_id in arcs:
        x1, y1 = coordinates[tail]
        x2, y2 = coordinates[head]
        linestyle = "-"  # if mode == 1 else '--'
        # Blue for mode 1 and green for mode 2 (if you want to use colors per route, swap accordingly)
        arccolor = (0.0, 1.0, 0.0) if mode == 2 else (0.0, 0.0, 1.0)
        ax.plot(
            [x1, x2],
            [y1, y2],
            linestyle=linestyle,
            color=arccolor,
            linewidth=4,
            zorder=1,
        )
    red = (1.0, 0.0, 0.0)
    for node, (x, y) in coordinates.items():
        if node == depot:
            ax.scatter(x, y, color=red, marker="s", s=60, zorder=2)
        else:
            ax.scatter(x, y, color=red, marker="o", s=60, zorder=2)
        # Optionally, you can uncomment the next line to add node labels:
        # ax.text(x + 0.1, y + 0.1, str(node), fontsize=9, color='blue')

    # Turn off the grid
    plt.grid(False)

    # Save the figure to a file and close the plot
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_all_solutions(
    arcs_folder,
    coordinates_folder,
    output_folder,
    bounds=(-1, 11, -1, 11),
    valid_range=None,
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in tqdm(os.listdir(arcs_folder), desc="Processing files", unit="file", leave=False):
        match = re.match(r"Arcs_(\w+)_\d+\.txt", filename)
        if not match:
             print(f"Skipped file (no match): {filename}")
        if match:
            number = int(match.group(1))
            if valid_range is None or number in valid_range:
                instance = match.group(1)
                print(instance)
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


@hydra.main(config_path="../../config/plot", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    numbers = cfg.numbers
    valid_range = range(cfg.valid_range[0], cfg.valid_range[1] + 1)
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
