import os
from ..graph import get_arc_name, get_coordinates_name, read_arcs, read_coordinates
from ..graph.HeatmapAnalyzer import HeatmapAnalyzer


def reverse_heatmap(cfg, fname: str, heatmap):
    """
    Reverse the heatmap to find arcs and coordinates that are in the zone defined by the heatmap. And then save them.
    Args:
        cfg: Configuration object containing paths and parameters.
        fname: The filename of the image.
        heatmap: The heatmap data to analyze.
    """

    number = int(fname.split(".")[0].split("_")[1])
    coordinates_p = os.path.join(cfg.arcs.coord_in_dir, get_coordinates_name(number))
    arcs_p = os.path.join(cfg.arcs.arcs_in_dir, get_arc_name(number))

    coordinates, _ = read_coordinates(coordinates_p, keep_service_time=True)
    arcs = read_arcs(arcs_p)

    analyzer = HeatmapAnalyzer(
        heatmap=heatmap,
        coordinates=coordinates,
        arcs=arcs,
        bounds=list(cfg.arcs.bounds),
        threshold=cfg.arcs.threshold,
        n_samples=cfg.arcs.n_samples,
    )
    arcs_with_zone, updated_coords = analyzer.reverse_heatmap()

    arcs_out = os.path.join(cfg.arcs.arcs_out_dir, get_arc_name(number))
    coords_out = os.path.join(cfg.arcs.coord_out_dir, get_coordinates_name(number))
    analyzer.write_arcs(arcs_with_zone, arcs_out)
    analyzer.write_coordinates(updated_coords, coords_out)
