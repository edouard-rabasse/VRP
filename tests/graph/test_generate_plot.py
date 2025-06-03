import tempfile
import os
import numpy as np
from src.graph.generate_plot import generate_plot_from_files, generate_plot_from_dict


def mock_arcs():
    return [
        (0, 1, 1, 0),
        (1, 2, 2, 0),
    ]


def mock_coordinates():
    return {
        0: (1.0, 1.0, 0),
        1: (2.0, 2.0, 0),
        2: (3.0, 1.0, 0),
    }


def test_generate_plot_from_dict():
    arcs = mock_arcs()
    coords = mock_coordinates()
    depot = 0

    img = generate_plot_from_dict(arcs, coords, depot)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3  # RGB
    assert img.shape[2] == 3  # 3 channels


def test_generate_plot_from_files():
    # Prepare fake arcs and coordinates files
    arcs_content = "0;1;1;0\n1;2;2;0\n"
    coords_content = "0,1.0,1.0,0\n1,2.0,2.0,0\n2,3.0,1.0,0\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        arcs_file = os.path.join(tmpdir, "arcs.txt")
        coords_file = os.path.join(tmpdir, "coords.txt")

        with open(arcs_file, "w") as f:
            f.write(arcs_content)
        with open(coords_file, "w") as f:
            f.write(coords_content)

        img = generate_plot_from_files(arcs_file, coords_file)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3
