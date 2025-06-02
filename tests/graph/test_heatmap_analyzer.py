import numpy as np
from src.graph.HeatmapAnalyzer import HeatmapAnalyzer


def mock_coordinates():
    return {
        0: (6.0, 9.0, 0),  # y inversé dans world_to_pixel
        1: (2.0, 3.0, 0),  # vertical arc (x = 2 constant)
        2: (4.0, 3.0, 0),  # horizontal arc
    }


def mock_arcs():
    return [
        (0, 1, 1, 0),  # vertical
        (1, 2, 2, 1),  # horizontal
    ]


def mock_heatmap():
    heatmap = np.zeros((10, 10))
    heatmap[2, 4] = 0.8  # pixel that vertical arc will cross
    heatmap[6, 3] = 0.7  # pixel that horizontal arc will cross
    return heatmap


def test_heatmap_analyzer_basic():
    heatmap = mock_heatmap()
    bounds = (0, 10, 0, 10)
    arcs = mock_arcs()
    coords = mock_coordinates()
    analyzer = HeatmapAnalyzer(
        heatmap, bounds, arcs, coords, threshold=0.5, n_samples=20
    )
    print(heatmap)

    in_zone = analyzer.arcs_in_zone()
    assert isinstance(in_zone, list)
    assert all(len(arc) == 4 for arc in in_zone)
    assert len(in_zone) > 0  # Should detect at least one arc
    assert len(in_zone) == 2  # Both arcs should be detected

    route_ids, points = analyzer.route_in_zone(in_zone)
    assert isinstance(route_ids, set)
    assert isinstance(points, set)

    arcs_flagged, coords_flagged = analyzer.reverse_heatmap()
    assert all(len(arc) == 5 for arc in arcs_flagged)
    assert all(len(val) == 4 for val in coords_flagged.values())
