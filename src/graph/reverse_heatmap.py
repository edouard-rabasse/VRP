import sys
from pathlib import Path

# Add the root directory to the system path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import numpy as np
from src.graph.coordinates import read_coordinates

## TODO: Add configuration file for the parameters
## TODO: Adapt to the mask


def read_coordinates(file_path):
    """Returns the coordinates file as a dictionary {node: (x, y)}."""
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


def get_arc_name(index):
    """Returns the arc name as a string."""
    return f"Arcs_{index}_1.txt"


def world_to_pixel(x, y, bounds, shape):
    """Convertit (x,y) réels ➜ indices de la matrice.
    bounds : (x_min, x_max, y_min, y_max)
    shape  : (n_rows, n_cols)"""
    x_min, x_max, y_min, y_max = bounds
    n_rows, n_cols = shape
    col = round((x - x_min) / (x_max - x_min) * (n_cols - 1))
    row = round((y_max - y) / (y_max - y_min) * (n_rows - 1))
    # Clip dans les bornes
    col = max(0, min(col, n_cols - 1))
    row = max(0, min(row, n_rows - 1))
    return row, col


def sample_segment(p1, p2, n=10):
    """n points équidistants entre p1 et p2 (inclus)."""
    x1, y1 = p1
    x2, y2 = p2
    for i in range(n):
        t = i / (n - 1)
        yield (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def is_arcs(arc, coordinates, heatmap, bounds, threshold=0.5, n_samples=15):
    """
    Retourne True si AU MOINS un point du segment
    traverse une cellule de heat‑mask > threshold.
    """
    tail, head, mode, route_id = arc
    p_tail = coordinates[tail]
    p_head = coordinates[head]

    # échantillonnage
    for x, y in sample_segment(p_tail, p_head, n_samples):
        r, c = world_to_pixel(x, y, bounds, heatmap.shape)
        if heatmap[r, c] >= threshold:
            return True  # inutile de tester d’autres points
    # si on arrive ici, c’est que le segment ne traverse pas de cellule de heat‑mask > threshold

    return (
        False  # aucun point du segment ne traverse une cellule de heat‑mask > threshold
    )


def arcs_in_zone(arcs, coordinates, heatmap, bounds, threshold=0.5, n_samples=15):
    """
    Retourne la liste des arcs dont AU MOINS un point du segment
    traverse une cellule de heat‑mask > threshold.
    """
    in_zone = []

    for arc in arcs:
        if is_arcs(arc, coordinates, heatmap, bounds, threshold, n_samples):

            # Ajouter l'arc à la liste avec les indices de la matrice
            in_zone.append(arc)

    return in_zone


def route_in_zone(arcs):
    """
    Returns the list of routes that have at least one point in the arcs.
    """
    route_ids = set(route_id for _, _, _, route_id in arcs)
    points = set(tail for tail, _, _, _ in arcs) | set(head for _, head, _, _ in arcs)
    return route_ids, points


def reverse_heatmap(arcs, coordinates, heatmap, bounds, threshold=0.5, n_samples=15):
    """
    Returns list of arcs and coordinates with added column equal to 1 if they are in the heatmap, else 0.
    """
    in_zone = arcs_in_zone(arcs, coordinates, heatmap, bounds, threshold, n_samples)
    route_ids, points = route_in_zone(in_zone)

    # Create a new list of arcs with the added column
    arcs_with_zone = []
    for arc in arcs:
        tail, head, mode, route_id = arc
        if arc in in_zone:
            arcs_with_zone.append((tail, head, mode, route_id, 1))  # In zone
        else:
            arcs_with_zone.append((tail, head, mode, route_id, 0))  # Not in zone
    for point in coordinates:
        if point in points:
            coordinates[point] = (coordinates[point][0], coordinates[point][1], 1)
        else:
            coordinates[point] = (coordinates[point][0], coordinates[point][1], 0)
    return arcs_with_zone, coordinates


### testing
if __name__ == "__main__":
    # Exemple d'utilisation
    # arcs = read_arcs("MSH/MSH/results/configuration1/Arcs_58_1.txt")
    # print(arcs)
    arcs = [(27, 44, 2, 2)]
    coordinates, _ = read_coordinates("MSH/MSH/instances/Coordinates_58.txt")

    heatmap = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3317561,
                0.76868343,
                0.45329687,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.4969085,
                1.0,
                0.49387857,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.13981317,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                0.02488444,
                0.00750479,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.13230991,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )  # Exemple de heatmap aléatoire
    bounds = (-1, 11, -1, 11)  # Exemple de bornes
    threshold = 0.2  # Exemple de seuil
    n_samples = 10  # Exemple de nombre d'échantillons
    in_zone = arcs_in_zone(arcs, coordinates, heatmap, bounds, threshold, n_samples)
    print("Arcs in zone:", in_zone)
    arcs_with_zone, coordinates = reverse_heatmap(
        arcs, coordinates, heatmap, bounds, threshold, n_samples
    )
    print("Coordinates with zone:", coordinates)
    print("Arcs with zone:", arcs_with_zone)
