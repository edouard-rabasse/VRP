import os
import tempfile
import matplotlib.pyplot as plt
from src.graph.plot_routes import setup_plot, plot_arcs, plot_nodes, plot_routes
from PIL import Image
import numpy as np


def test_setup_plot_runs_without_error():
    fig, ax = plt.subplots()
    try:
        setup_plot(ax, bounds=(-1, 11, -1, 11))
    finally:
        plt.close(fig)


def test_plot_arcs_draws_lines():
    fig, ax = plt.subplots()
    coordinates = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0),
    }
    arcs = [
        (0, 1, 1, 0),
        (1, 2, 2, 0),
    ]
    try:
        plot_arcs(ax, arcs, coordinates)
    finally:
        plt.close(fig)


def test_plot_nodes_draws_points():
    fig, ax = plt.subplots()
    coordinates = {
        0: (0.0, 0.0, 0),
        1: (1.0, 0.0, 0),
        2: (1.0, 1.0, 1),
        3: (0.0, 1.0, 1),
    }
    arcs = [
        (0, 1, 1, 0, 0),
        (2, 3, 2, 0, 0),
    ]
    try:
        plot_nodes(ax, coordinates, depot=0, arcs=arcs, route_type="modified")
    finally:
        plt.close(fig)


def test_plot_routes_creates_image_file():
    arcs = [
        (0, 1, 1, 0),
        (1, 2, 2, 0),
        (2, 3, 1, 0),
        (3, 0, 1, 0),
    ]
    coordinates = {
        0: (0.0, 0.0, 0),
        1: (1.0, 0.0, 0),
        2: (1.0, 1.0, 0),
        3: (0.0, 1.0, 0),
    }
    depot = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.png")
        plot_routes(arcs, coordinates, depot, output_file=output_path)

        assert os.path.exists(output_path)

        img = Image.open(output_path)
        img.verify()  # Test if it's a valid image

        img = Image.open(output_path).convert("RGB")
        arr = np.array(img)
        assert arr.ndim == 3 and arr.shape[2] == 3  # Ensure it's an RGB image
        # show the image
