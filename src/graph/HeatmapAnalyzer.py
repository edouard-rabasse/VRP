# HeatmapAnalyzer.py
"""
HeatmapAnalyzer module for analyzing arcs and coordinates with respect to a spatial heatmap.
Classes:
    HeatmapAnalyzer: Analyzes a heatmap to determine which arcs and coordinates fall within specified zones.
Functions:

    __init__(self, heatmap, arcs, coordinates, bounds=(-1, 11, -1, 11), threshold=0.5, n_samples=15)
        Initializes the HeatmapAnalyzer with the given heatmap, arcs, coordinates, spatial bounds, threshold, and number of samples.
    world_to_pixel(self, x, y)
        Converts world coordinates (x, y) to pixel indices in the heatmap array.
    sample_segment(self, p1, p2)
        Generates sample points along the segment between two points p1 and p2.
    is_arc_in_zone(self, arc)
        Determines if any part of the given arc passes through a zone in the heatmap above the threshold.
    arcs_in_zone(self)
        Returns a list of arcs that pass through zones in the heatmap above the threshold.
    route_in_zone(self, arcs_in_zone)
        Returns the set of route IDs and points that are part of the arcs in the zone.
    reverse_heatmap(self, return_weighted_sum=False, top_n_arcs=None)
        Determines which arcs and coordinates are in the zone and returns them with zone indicators.
    write_arcs(self, arcs_with_zone, output_path)
        Writes the arcs with zone indicators to a specified output file.
    write_coordinates(self, coordinates, output_path)
        Writes the coordinates with zone indicators to a specified output file.

Usage Example:

    Instantiate HeatmapAnalyzer with a heatmap, arcs, and coordinates, then use its methods to analyze and export results.

"""


import sys
from pathlib import Path
import os

# Add the root directory to the system path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import numpy as np
from src.graph.read_coordinates import read_coordinates, get_coordinates_name
from src.graph.read_arcs import read_arcs, get_arc_name


## TODO: Add configuration file for the parameters
## TODO: Adapt to the mask
class HeatmapAnalyzer:
    def __init__(
        self,
        heatmap,
        arcs,
        coordinates,
        bounds=(-1, 11, -1, 11),
        threshold=0.5,
        n_samples=15,
        return_weighted_sum=False,
    ):
        """Initialize the HeatmapAnalyzer.

        Args:
            heatmap (np.ndarray): The heatmap to analyze.
            bounds (tuple): The spatial bounds of the heatmap (x_min, x_max, y_min, y_max).
            arcs (list): The list of arcs to consider.
            coordinates (dict): The dictionary of node coordinates.
            threshold (float, optional): The threshold for considering an arc important. Defaults to 0.5.
            n_samples (int, optional): The number of samples to take along each arc. Defaults to 15.
        """
        self.heatmap = heatmap
        self.bounds = bounds
        self.arcs = arcs
        self.coordinates = coordinates
        self.threshold = threshold
        self.n_samples = n_samples
        self.return_weighted_sum = return_weighted_sum

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

    def is_arc_in_zone(self, arc, return_weighted_sum=False):
        tail, head = arc[:2]
        p_tail = self.coordinates[tail][:2]
        p_head = self.coordinates[head][:2]
        values = []
        for x, y in self.sample_segment(p_tail, p_head):
            r, c = self.world_to_pixel(x, y)
            value = self.heatmap[r, c]
            if not return_weighted_sum:
                if value >= self.threshold:
                    return 1
            else:
                values.append(value)
        if return_weighted_sum:
            return sum(values) / len(values) if values else 0.0
        return 0

    def arcs_in_zone(self, return_weighted_sum=False):
        return [
            arc
            for arc in self.arcs
            if self.is_arc_in_zone(arc, return_weighted_sum=return_weighted_sum)
        ]

    def route_in_zone(self, arcs_in_zone):
        route_ids = set(arc[3] for arc in arcs_in_zone)
        points = set(arc[0] for arc in arcs_in_zone) | set(
            arc[1] for arc in arcs_in_zone
        )
        return route_ids, points

    def reverse_heatmap(self, return_weighted_sum=False, top_n_arcs=None):
        """Reverse the heatmap to find arcs and coordinates in the zone.

        Args:
            return_weighted_sum (bool): If True, return the weighted sum for each arc instead of a binary flag.
            top_n_arcs (int or None): If set and return_weighted_sum is True, only flag the top n arcs by weighted sum.

        Returns:
            arcs: List of arcs with an indicator (0/1 or weighted sum) of whether they are in the zone.
            coordinates: Dictionary of coordinates with an indicator (0/1 or weighted sum) of whether they are in the zone.
        """
        arc_values = [
            self.is_arc_in_zone(arc, return_weighted_sum=return_weighted_sum)
            for arc in self.arcs
        ]

        if return_weighted_sum and top_n_arcs is not None:
            # Get indices of top n arcs by weighted sum
            sorted_indices = sorted(
                range(len(arc_values)), key=lambda i: arc_values[i], reverse=True
            )
            top_indices = set(sorted_indices[:top_n_arcs])
            # Set value to weighted sum for top n, 0 for others
            filtered_arc_values = [
                arc_values[i] if i in top_indices else 0 for i in range(len(arc_values))
            ]
        else:
            filtered_arc_values = arc_values

        # For arcs: attach the value (weighted sum or 0/1)
        arcs_with_flag = [
            (*arc[:4], value) for arc, value in zip(self.arcs, filtered_arc_values)
        ]

        # For coordinates: aggregate the values of arcs that include each point
        coord_values = {point: [] for point in self.coordinates}
        for arc, value in zip(self.arcs, filtered_arc_values):
            tail, head = arc[:2]
            coord_values[tail].append(value)
            coord_values[head].append(value)
        new_coordinates = {}
        for point in self.coordinates:
            if coord_values[point]:
                # Use max value among all incident arcs
                agg_value = max(coord_values[point])
            else:
                agg_value = 0
            new_coordinates[point] = (*self.coordinates[point][:3], agg_value)
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
