"""
This module provides functions to generate a plot of routes from VRP (Vehicle Routing Problem) data files or in-memory data structures.

It leverages the `plot_routes` function to visualize routes based on arcs and coordinates, and returns the resulting plot as a numpy array.
Functions:

    - generate_plot_from_files(
        Reads arcs and coordinates from specified files, generates a plot of the routes, and returns the image as a numpy array.
            arcs_file (str): Path to the arcs file.
            coords_file (str): Path to the coordinates file.
            route_type (str, optional): Type of route to plot. Defaults to "original".
            show_labels (bool, optional): Whether to show node labels. Defaults to False.
            background_image (optional): Background image for the plot. Defaults to None.
            np.ndarray: Image as a numpy array.

    - generate_plot_from_dict(
        Generates a plot from a list of arcs and a dictionary of coordinates, and returns the image as a numpy array.
            arcs (list): List of arcs representing the routes.
            coordinates (dict): Dictionary mapping node identifiers to coordinates.
            depot (int): Depot node identifier.
            route_type (str, optional): Type of route to plot. Defaults to "original".
            show_labels (bool, optional): Whether to show node labels. Defaults to False.
            background_image (optional): Background image for the plot. Defaults to None.
            np.ndarray: Image as a numpy array.
"""

import io
from .read_coordinates import read_coordinates
from .read_arcs import read_arcs
from .plot_routes import plot_routes
from typing import Tuple, Dict, List
import numpy as np
from PIL import Image


def generate_plot_from_files(
    arcs_file: str,
    coords_file: str,
    bounds=(-1, 11, -1, 11),
    dpi=100,
    route_type="original",
    show_labels=False,
    background_image=None,
) -> np.ndarray:
    """
    Generate a plot from arcs and coordinates files and return as a numpy array.
    Uses existing plot_routes function from graph_creator.

    Args:
        arcs_file: Path to the arcs file
        coords_file: Path to the coordinates file
        bounds: Plot bounds (x_min, x_max, y_min, y_max)
        dpi: DPI for the plot (affects output resolution)

    Returns:
        tuple: (image_array, coordinates_dict, arcs_list, depot_node)
    """
    # Read arcs and coordinates
    arcs = read_arcs(arcs_file, type=route_type)
    coordinates, depot = read_coordinates(coords_file)

    img = generate_plot_from_dict(
        arcs=arcs,
        coordinates=coordinates,
        depot=depot,
        bounds=bounds,
        dpi=dpi,
        route_type=route_type,
        show_labels=show_labels,
        background_image=background_image,
    )
    return img


def generate_plot_from_dict(
    arcs,
    coordinates,
    depot,
    bounds=(-1, 11, -1, 11),
    dpi=100,
    route_type="original",
    show_labels=False,
    background_image=None,
) -> np.ndarray:
    """Generate a plot from list of arcs and dictionnary of coordinates and return as a numpy array.

    Args:
        arcs (list): List of arcs
        coordinates (dict): Dictionary of coordinates
        depot (int): Depot node identifier
        bounds (tuple, optional): Plot bounds (x_min, x_max, y_min, y_max). Defaults to (-1, 11, -1, 11).
        dpi (int, optional): DPI for the plot (affects output resolution). Defaults to 100.

    Returns:
        np.ndarray: Image as a numpy array
    """

    # Create in-memory buffer for the image
    buf = io.BytesIO()

    # Use the existing plot_routes function to generate the plot
    plot_routes(
        arcs,
        coordinates,
        depot,
        buf,
        bounds=bounds,
        route_type=route_type,
        show_labels=show_labels,
        background_image=background_image,
    )

    # Convert buffer to numpy array
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    buf.close()

    return img
