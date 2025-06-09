# pipeline.py: Takes arcs and coordinates, processes into an image, and uses a model to highlight important arcs/nodes

import os
import io
import numpy as np
import torch
import matplotlib
from hydra import initialize, compose
import time

matplotlib.use("Agg")  # Use non-interactive backend to avoid memory issues

import argparse

# Import existing graph creator functions
from src.graph import read_coordinates, read_arcs
from src.graph.graph_flagging import flag_graph_from_data
from src.models import load_model


device = "cuda" if torch.cuda.is_available() else "cpu"


with initialize(version_base=None, config_path="config"):
    cfg = compose(
        config_name="config",
        overrides=[
            "data=config7",
            "model=resnet",
            "model.weight_path=checkpoints/resnet_8_30_7.pth",
        ],
    )


def initialise_model():
    model = load_model(
        cfg.model.name,
        "cuda" if torch.cuda.is_available() else "cpu",
        cfgm=cfg.model,
    )
    dummy_input = torch.zeros(1, 3, 224, 224).to(
        device
    )  # Batch de taille 1 pour une image 224x224
    model(dummy_input)
    return model


def pipeline():

    model = initialise_model()

    number = 6

    time_start = time.perf_counter()

    coord_path = f"MSH/MSH/instances/Coordinates_{number}.txt"
    arc_path = f"MSH/MSH/results/configuration1/Arcs_{number}_1.txt"
    modified_arcs_path = f"MSH/MSH/results/configuration7/Arcs_{number}_1.txt"
    arcs = read_arcs(arc_path)
    coordinates, depot = read_coordinates(coord_path)

    flagged_arcs, flagged_coordinates = flag_graph_from_data(
        arcs,
        coordinates,
        depot,
        model,
        cfg,
        device=device,
    )
    time_end = time.perf_counter()
    print(f"[Pipeline] Total processing time: {time_end - time_start:.2f} seconds")
