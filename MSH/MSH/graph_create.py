import matplotlib.pyplot as plt
import numpy as np
import os
import re
from tqdm import tqdm

def read_arcs(file_path):
    arcs = []
    with open(file_path, 'r') as file:
        for line in file:
            tail, head, mode, route_id = map(int, line.strip().split(';'))
            arcs.append((tail, head, mode, route_id))
    return arcs

def read_coordinates(file_path):
    coordinates = {}
    last_node = None
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            node = int(parts[0])
            x, y = map(float, parts[1:3])
            coordinates[node] = (x, y)
            last_node = node  # The last node is the depot
    return coordinates, last_node


def plot_routes(arcs, coordinates, depot, output_file):
    # Create a figure and axes with a 10x10 inch size and equal aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='box')
    
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
        linestyle = '-' #if mode == 1 else '--'
        # Blue for mode 1 and green for mode 2 (if you want to use colors per route, swap accordingly)
        arccolor = "green" if mode == 2 else "blue"
        ax.plot([x1, x2], [y1, y2], linestyle=linestyle, color=arccolor, linewidth=6, zorder=1)
    
    # Plot nodes with different markers: red squares for the depot, black circles for other nodes
    for node, (x, y) in coordinates.items():
        if node == depot:
            ax.scatter(x, y, color='red', marker='s', s = 200,zorder=2)
        else:
            ax.scatter(x, y, color='black', marker='o', s = 200, zorder=2)
        # Optionally, you can uncomment the next line to add node labels:
        # ax.text(x + 0.1, y + 0.1, str(node), fontsize=9, color='blue')
    
    # Turn off the grid
    plt.grid(False)
    
    # Save the figure to a file and close the plot
    plt.savefig(output_file)
    plt.close()


def process_all_solutions(arcs_folder, coordinates_folder, output_folder,configuration_number=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in tqdm(os.listdir(arcs_folder), desc=f"Processing configuration {configuration_number}", unit="file"):
        match = re.match(r'Arcs_(\w+)_\d+\.txt', filename)
        if match:
            instance = match.group(1)
            arcs_file = os.path.join(arcs_folder, filename)
            coordinates_file = os.path.join(coordinates_folder, f'Coordinates_{instance}.txt')
            output_file = os.path.join(output_folder, f'Plot_{instance}.png')
            
            if os.path.exists(coordinates_file):
                arcs = read_arcs(arcs_file)
                coordinates, depot = read_coordinates(coordinates_file)
                plot_routes(arcs, coordinates, depot, output_file)
            else:
                print(f"Warning: Coordinates file {coordinates_file} not found.")



if __name__ == "__main__":
    # Define the configuration numbers to process
    # You can change this list to include any configuration numbers you want to process

    numbers = range(1,8)
    for number in numbers:
        arcs_folder = f"MSH/MSH/results/configuration{number}/"
        coordinates_folder = "MSH/MSH/instances/"
        output_folder = f"MSH/MSH/plots/configuration{number}/"

        # Process all solutions
        process_all_solutions(arcs_folder, coordinates_folder, output_folder)





