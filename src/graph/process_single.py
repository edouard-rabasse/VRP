from .read_arcs import read_arcs
from .read_coordinates import read_coordinates
from .plot_routes import plot_routes


def process_single_solution(
    arcs_file,
    coordinates_file,
    output_file,
    bounds,
    route_type,
    show_labels,
    background_image=None,
):
    arcs = read_arcs(arcs_file, type=route_type)
    coordinates, depot = read_coordinates(coordinates_file, type=route_type)
    plot_routes(
        arcs,
        coordinates,
        depot,
        output_file,
        bounds,
        route_type,
        show_labels,
        background_image,
    )
