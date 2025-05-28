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
from src.graph.graph_creator import read_arcs, read_coordinates, plot_routes
from src.models import load_model
from src.visualization import get_heatmap
from src.transform import image_transform_test


def generate_plot_from_arcs(
    arcs_file: str, coords_file: str, bounds=(-1, 11, -1, 11), dpi=100
) -> Tuple[np.ndarray, Dict, List, int]:
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
    arcs = read_arcs(arcs_file)
    coordinates, depot = read_coordinates(coords_file)

    # Create in-memory buffer for the image
    buf = io.BytesIO()

    # Use the existing plot_routes function to generate the plot
    plot_routes(arcs, coordinates, depot, buf, bounds=bounds)

    # Convert buffer to numpy array
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    buf.close()

    return img, coordinates, arcs, depot


def extract_important_arcs(
    image: np.ndarray,
    arcs: List,
    coordinates: Dict,
    model_path: str,
    model_name: str,
    heatmap_method: str,
    heatmap_args: Dict,
    threshold: float = 0.7,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List:
    """
    Extract important arcs based on model heatmap.

    Args:
        image: Input image (numpy array)
        arcs: List of arcs (tail, head, mode, route_id)
        coordinates: Dict of node coordinates
        model_path: Path to the model weights
        model_name: Name of the model architecture
        heatmap_method: Method for generating heatmap
        heatmap_args: Arguments for heatmap generation
        threshold: Threshold for considering an arc important
        device: Device to run the model on

    Returns:
        list: List of important arcs
    """
    # Load model
    model = load_model(model_name, device, {"weight_path": model_path})
    model.eval()

    # Prepare image for model
    transform = image_transform_test()
    img_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)

    # Generate heatmap
    heatmap = get_heatmap(heatmap_method, model, img_tensor, heatmap_args, device)

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Determine important arcs by checking if they pass through high-activation areas
    important_arcs = []
    for arc in arcs:
        tail, head, mode, route_id = arc
        x1, y1 = coordinates[tail]
        x2, y2 = coordinates[head]

        # Convert from coordinate space to pixel space
        # This mapping depends on how the image was rendered and needs to be calibrated
        h, w = image.shape[:2]
        px1 = int((x1 - bounds[0]) / (bounds[1] - bounds[0]) * w)
        py1 = int((y1 - bounds[2]) / (bounds[3] - bounds[2]) * h)
        px2 = int((x2 - bounds[0]) / (bounds[1] - bounds[0]) * w)
        py2 = int((y2 - bounds[2]) / (bounds[3] - bounds[2]) * h)

        # Sample points along the arc
        num_samples = 20
        xs = np.linspace(px1, px2, num_samples).astype(int)
        ys = np.linspace(py1, py2, num_samples).astype(int)

        # Clip to image boundaries
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)

        # Check if any point along the arc has high activation
        max_activation = max(heatmap_resized[ys[i], xs[i]] for i in range(num_samples))

        if max_activation > threshold:
            important_arcs.append(arc)

    return important_arcs


def write_modified_arcs(original_arcs: List, important_arcs: List, output_file: str):
    """
    Write modified arcs to output file, with important arcs marked with mode=3.

    Args:
        original_arcs: List of original arcs
        important_arcs: List of important arcs to highlight
        output_file: Path to write the output file
    """
    important_set = set((t, h) for t, h, _, _ in important_arcs)

    with open(output_file, "w") as f:
        for tail, head, mode, route_id in original_arcs:
            # Mark important arcs with mode=3
            if (tail, head) in important_set:
                mode = 3  # Special mode for highlighted arcs

            f.write(f"{tail};{head};{mode};{route_id}\n")


def main():
    parser = argparse.ArgumentParser(description="VRP route analysis pipeline")
    parser.add_argument("--arcs", required=True, help="Path to arcs file")
    parser.add_argument("--coords", required=True, help="Path to coordinates file")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--model-name", default="vgg", help="Model architecture name")
    parser.add_argument("--output", required=True, help="Path to output arcs file")
    parser.add_argument(
        "--heatmap-method", default="grad_cam_vgg", help="Heatmap method"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Threshold for important arcs"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save visualization of results"
    )
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        default=[-1, 11, -1, 11],
        help="Plot bounds: x_min x_max y_min y_max",
    )

    args = parser.parse_args()

    # Set global bounds for coordinate to pixel conversion
    global bounds
    bounds = tuple(args.bounds)

    # Define heatmap arguments based on the method
    if args.heatmap_method == "grad_cam_vgg":
        heatmap_args = {"class_index": 1}  # Modified routes class
    elif args.heatmap_method == "grad_rollout":
        heatmap_args = {"class_index": 1, "discard_ratio": 0.9}
    else:
        heatmap_args = {"class_index": 1}  # Default

    # Generate plot from arcs and coordinates
    print(f"Generating plot from {args.arcs} and {args.coords}")
    image, coordinates, arcs, depot = generate_plot_from_arcs(
        args.arcs, args.coords, bounds=bounds
    )

    # Extract important arcs using the model
    print(f"Analyzing plot with model {args.model}")
    important_arcs = extract_important_arcs(
        image,
        arcs,
        coordinates,
        args.model,
        args.model_name,
        args.heatmap_method,
        heatmap_args,
        args.threshold,
    )

    # Write modified arcs to output file
    print(f"Writing {len(important_arcs)} highlighted arcs to {args.output}")
    write_modified_arcs(arcs, important_arcs, args.output)

    # Optionally save visualization
    if args.visualize:
        vis_path = args.output.replace(".txt", "_visualization.png")
        print(f"Saving visualization to {vis_path}")

        # Create visualization with highlighted arcs
        # Mark highlighted arcs with special mode=4 for visualization
        visualize_arcs = []
        highlight_set = set((t, h) for t, h, _, _ in important_arcs)

        for tail, head, mode, route_id in arcs:
            if (tail, head) in highlight_set:
                # Use mode=4 for highlighting in red
                visualize_arcs.append((tail, head, 4, route_id))
            else:
                visualize_arcs.append((tail, head, mode, route_id))

        # Call plot_routes with the modified arcs
        plot_routes(visualize_arcs, coordinates, depot, vis_path, bounds=bounds)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
