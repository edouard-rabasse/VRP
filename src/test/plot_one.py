import sys
from pathlib import Path

# Add the root directory to the system path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
import argparse
import matplotlib.pyplot as plt
from src.graph.generate_plot import generate_plot_from_files


def parse_arguments():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Generate a plot from arcs and coordinates files"
    )

    # Add argument for arcs file
    parser.add_argument(
        "-a", "--arcs", type=str, required=True, help="Path to the arcs file"
    )

    # Add argument for coordinates file
    parser.add_argument(
        "-c", "--coords", type=str, required=True, help="Path to the coordinates file"
    )

    # Optional argument for bounds (default to (-1, 11, -1, 11))
    parser.add_argument(
        "-b",
        "--bounds",
        type=str,
        default="(-1, 11, -1, 11)",
        help="Bounds for the plot in the format (x_min, x_max, y_min, y_max)",
    )

    # Parse the arguments
    return parser.parse_args()


def main():
    # Parse the command-line arguments
    args = parse_arguments()

    # Convert the bounds argument from string to tuple
    bounds = tuple(map(int, args.bounds.strip("()").split(",")))

    # Generate the plot using the provided files and bounds
    img_array = generate_plot_from_files(args.arcs, args.coords, bounds=bounds)

    # Display the image using matplotlib
    plt.imshow(img_array)
    plt.axis("off")  # Hide axes
    plt.show()  # Show the plot


if __name__ == "__main__":
    main()
