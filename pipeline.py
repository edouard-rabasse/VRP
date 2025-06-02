# pipeline.py: Takes arcs and coordinates, processes into an image, and uses a model to highlight important arcs/nodes

import os
import io
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid memory issues
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
from typing import List, Tuple, Dict, Set
from torch.utils.data import DataLoader
from torchvision import transforms

# Import existing graph creator functions
from src.graph import generate_plot_from_files, read_coordinates, read_arcs
from src.models import load_model
from src.visualization import get_heatmap
from src.transform import image_transform_test

from src.graph import HeatmapAnalyzer


coord_path = "MSH/MSH/instances/Coordinates_5.txt"
arc_path = "MSH/MSH/resulats/configuration1/Arcs_5_1.txt"
modified_arcs_path = "MSH/MSH/resulats/configuration7/Arcs_5_1.txt"

model = (
    load_model(
        "resnet",
        "cuda" if torch.cuda.is_available() else "cpu",
        {"weight_path": "checkpoints/resnet_8_30_7.pth"},
    ),
)

original_img = generate_plot_from_files(
    arcs_file=arc_path, coords_file=coord_path, bounds=(-1, 11, -1, 11)
)

modified_img = generate_plot_from_files(
    arcs_file=modified_arcs_path, coords_file=coord_path, bounds=(-1, 11, -1, 11)
)


heatmap = get_heatmap(
    method="grad_cam",
    img_tensor=image_transform_test()(Image.fromarray(original_img))
    .unsqueeze(0)
    .to("cuda" if torch.cuda.is_available() else "cpu"),
    args={"class_index": 1},
    device="cuda" if torch.cuda.is_available() else "cpu",
)
arcs = read_arcs(arc_path)
coordinates, _ = read_coordinates(coord_path)

hm_analyzer = HeatmapAnalyzer(heatmap=heatmap, arcs=arcs, coordinates=coordinates)

arcs_flagged, coord_flagged = hm_analyzer.reverse_heatmap()
