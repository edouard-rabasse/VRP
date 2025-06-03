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

from omegaconf import OmegaConf


device = "cuda" if torch.cuda.is_available() else "cpu"

cfgm = OmegaConf.create(
    {
        "weight_path": "checkpoints/resnet_8_30_7.pth",
        "kernel_size": 7,
        "freeze": False,
        "load": True,
    }
)

model = load_model(
    "resnet",
    "cuda" if torch.cuda.is_available() else "cpu",
    cfgm=cfgm,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process graph data and visualize heatmaps."
    )

    number = 6

    coord_path = f"MSH/MSH/instances/Coordinates_{number}.txt"
    arc_path = f"MSH/MSH/results/configuration1/Arcs_{number}_1.txt"
    modified_arcs_path = f"MSH/MSH/results/configuration7/Arcs_{number}_1.txt"

    original_img = generate_plot_from_files(
        arcs_file=arc_path, coords_file=coord_path, bounds=(-1, 11, -1, 11)
    )
    img_tensor = image_transform_test()(Image.fromarray(original_img))
    img_tensor = img_tensor.unsqueeze(0).to(device)

    proba_of_needing_modif = torch.sigmoid(model(img_tensor)).squeeze()[1].item()
    print(f"Probability of needing modification: {proba_of_needing_modif:.4f}")

    modified_img = generate_plot_from_files(
        arcs_file=modified_arcs_path, coords_file=coord_path, bounds=(-1, 11, -1, 11)
    )

    args = OmegaConf.create(
        {
            "class_index": 1,  # Assuming class index 1 is the one of interest
            "target_layer": "backbone.layer4.1.conv2",  # Adjust based on your model architecture
            "discard_ratio": 0.9,  # Example value for Grad Rollout
        }
    )

    heatmap = get_heatmap(
        model=model,
        method="gradcam",
        input_tensor=img_tensor,
        args=args,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    arcs = read_arcs(arc_path)
    coordinates, _ = read_coordinates(coord_path)

    hm_analyzer = HeatmapAnalyzer(heatmap=heatmap, arcs=arcs, coordinates=coordinates)

    arcs_flagged, coord_flagged = hm_analyzer.reverse_heatmap()
