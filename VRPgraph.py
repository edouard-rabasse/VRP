import os
import math
from collections import defaultdict
from typing import List, Tuple
from torch.utils.data import Dataset


class VRPGraphInstance:
    def __init__(self, arcs_file: str, coordinates_file: str):
        self.arcs_file = arcs_file
        self.coordinates_file = coordinates_file
        self.arcs = self.read_arcs()
        self.coordinates, self.depot = self.read_coordinates()
        self.routes = self.group_arcs_by_route()
        self.features = self.compute_features()

    def read_arcs(self) -> List[Tuple[int, int, int, int]]:
        arcs = []
        with open(self.arcs_file, "r") as file:
            for line in file:
                tail, head, mode, route_id = map(int, line.strip().split(";"))
                arcs.append((tail, head, mode, route_id))
        return arcs

    def read_coordinates(self) -> Tuple[dict, int]:
        coordinates = {}
        last_node = None
        with open(self.coordinates_file, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                node = int(parts[0])
                x, y = map(float, parts[1:3])
                coordinates[node] = (x, y)
                last_node = node  # The last node is the depot
        return coordinates, last_node

    def group_arcs_by_route(self):
        routes = defaultdict(list)
        for tail, head, mode, route_id in self.arcs:
            routes[route_id].append((tail, head, mode))
        return routes

    def compute_distance(self, node1: int, node2: int) -> float:
        x1, y1 = self.coordinates[node1]
        x2, y2 = self.coordinates[node2]
        return math.hypot(x2 - x1, y2 - y1)

    def compute_features(self):
        features = {
            "number_of_routes": len(self.routes),
            "average_points_per_route": 0.0,
            "walked_distance_per_route": [],
            "max_walked_segment": 0.0,
        }

        total_points = 0
        max_walked_segment = 0.0

        for route_id, arcs in self.routes.items():
            point_set = set()
            walked_distance = 0.0
            for tail, head, mode in arcs:
                point_set.update([tail, head])
                if mode == 2:  # walking mode
                    dist = self.compute_distance(tail, head)
                    walked_distance += dist
                    if dist > max_walked_segment:
                        max_walked_segment = dist
            total_points += len(point_set)
            features["walked_distance_per_route"].append(walked_distance)

        features["average_points_per_route"] = (
            total_points / len(self.routes) if self.routes else 0
        )
        features["max_walked_segment"] = max_walked_segment
        return features

    def get_features(self):
        return self.features


class VRPGraphDataset(Dataset):
    def __init__(
        self,
        arcs_original_folder: str,
        arcs_modified_folder: str,
        coordinates_folder: str,
    ):
        self.instances = []  # list of (arcs_path, coords_path, label)
        self.names = []

        # Original instances (label 0)
        for file in os.listdir(arcs_original_folder):
            if file.startswith("Arcs_") and file.endswith(".txt"):
                instance_id = file[len("Arcs_") : -len("_1.txt")]
                print(instance_id)
                coords_file = f"Coordinates_{instance_id}.txt"
                arcs_path = os.path.join(arcs_original_folder, file)
                coords_path = os.path.join(coordinates_folder, coords_file)
                if os.path.exists(coords_path):
                    print("a")
                    self.instances.append((arcs_path, coords_path, 0))
                    self.names.append(instance_id)

        # Modified instances (label 1)
        for file in os.listdir(arcs_modified_folder):
            if file.startswith("Arcs_") and file.endswith(".txt"):
                instance_id = file[len("Arcs_") : -len(".txt")]
                coords_file = f"Coordinates_{instance_id}.txt"
                arcs_path = os.path.join(arcs_modified_folder, file)
                coords_path = os.path.join(coordinates_folder, coords_file)
                if os.path.exists(coords_path):
                    self.instances.append((arcs_path, coords_path, 1))
                    self.names.append(instance_id)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        arcs_path, coords_path, label = self.instances[idx]
        instance = VRPGraphInstance(arcs_path, coords_path)
        features = instance.get_features()
        return features, label


if __name__ == "__main__":
    # Example usage
    coordinates_folder_path = "MSH/MSH/instances/"
    arcs_modified = "MSH/MSH/results/configuration7/"
    arcs_original = "MSH/MSH/results/configuration1/"
    dataset = VRPGraphDataset(arcs_original, arcs_modified, coordinates_folder_path)
    for i in range(len(dataset)):
        features = dataset[i]
        print(f"Instance {i}: {features}")
        print(f"Instance name: {dataset.names[i]}")
        print("-" * 20)
