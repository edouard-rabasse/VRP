import sys
from pathlib import Path
import os

# Add the root directory to the system path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import numpy as np
from src.graph.coordinates import read_coordinates, get_coordinates_name
from src.graph.arcs import read_arcs, get_arc_name


## TODO: Add configuration file for the parameters
## TODO: Adapt to the mask
class HeatmapAnalyzer:
    def __init__(self, heatmap, bounds, arcs, coordinates, threshold=0.5, n_samples=15):
        self.heatmap = heatmap
        self.bounds = bounds
        self.arcs = arcs
        self.coordinates = coordinates
        self.threshold = threshold
        self.n_samples = n_samples

    def world_to_pixel(self, x, y):
        x_min, x_max, y_min, y_max = self.bounds
        n_rows, n_cols = self.heatmap.shape
        col = round((x - x_min) / (x_max - x_min) * (n_cols - 1))
        row = round((y_max - y) / (y_max - y_min) * (n_rows - 1))
        return max(0, min(row, n_rows - 1)), max(0, min(col, n_cols - 1))

    def sample_segment(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        for i in range(self.n_samples):
            t = i / (self.n_samples - 1)
            yield (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def is_arc_in_zone(self, arc):
        tail, head, _, _ = arc
        p_tail = self.coordinates[tail][:2]
        p_head = self.coordinates[head][:2]
        for x, y in self.sample_segment(p_tail, p_head):
            r, c = self.world_to_pixel(x, y)
            if self.heatmap[r, c] >= self.threshold:
                return True
        return False

    def arcs_in_zone(self):
        return [arc for arc in self.arcs if self.is_arc_in_zone(arc)]

    def route_in_zone(self, arcs_in_zone):
        route_ids = set(route_id for _, _, _, route_id in arcs_in_zone)
        points = set(tail for tail, _, _, _ in arcs_in_zone) | set(
            head for _, head, _, _ in arcs_in_zone
        )
        return route_ids, points

    def reverse_heatmap(self):
        in_zone = self.arcs_in_zone()
        route_ids, points = self.route_in_zone(in_zone)

        arcs_with_flag = [(*arc, 1 if arc in in_zone else 0) for arc in self.arcs]

        new_coordinates = {}

        for point in self.coordinates:
            if point in points:
                new_coordinates[point] = (*self.coordinates[point][:3], 1)
            else:
                new_coordinates[point] = (*self.coordinates[point][:3], 0)

        return arcs_with_flag, new_coordinates

    def write_arcs(self, arcs_with_zone, output_path):
        """Écrit les arcs dans un fichier.
        Args:
            arcs_with_zone (list): Liste des arcs avec un indicateur de zone.
            output_path (str): Chemin du fichier de sortie.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for arc in arcs_with_zone:
                f.write(f"{arc[0]};{arc[1]};{arc[2]};{arc[3]};{arc[4]}\n")

    def write_coordinates(self, coordinates, output_path):
        """Écrit les coordonnées dans un fichier.
        Args:
            coordinates (dict): Dictionnaire des coordonnées avec un indicateur de zone.
            output_path (str): Chemin du fichier de sortie."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for node, coord in coordinates.items():
                f.write(f"{node},{coord[0]},{coord[1]},{coord[2]},{coord[3]}\n")


### testing
if __name__ == "__main__":
    # Exemple d'utilisation
    # arcs = read_arcs("MSH/MSH/results/configuration1/Arcs_58_1.txt")
    # print(arcs)
    arcs = [(27, 44, 2, 2)]
    coordinates, _ = read_coordinates(
        "MSH/MSH/instances/Coordinates_58.txt", keep_service_time=True
    )

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
    Hm_analyzer = HeatmapAnalyzer(
        heatmap, bounds, arcs, coordinates, threshold=threshold, n_samples=n_samples
    )
    in_zone = Hm_analyzer.arcs_in_zone()
    print("Arcs in zone:", in_zone)
    arcs_with_zone, coordinates = Hm_analyzer.reverse_heatmap()
    print("Coordinates with zone:", coordinates)
    print("Arcs with zone:", arcs_with_zone)
