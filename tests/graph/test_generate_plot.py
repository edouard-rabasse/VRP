import tempfile
import os
import numpy as np
from src.graph.generate_plot_from_files import generate_plot_from_files


def test_generate_plot_from_files():
    # Contenu minimal de fichiers arcs et coordinates
    arcs_content = "0;1;1;0\n1;2;2;0\n"
    coords_content = "0,0,0\n1,1,0\n2,2,0\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        arcs_path = os.path.join(tmpdir, "arcs.txt")
        coords_path = os.path.join(tmpdir, "coords.txt")

        with open(arcs_path, "w") as f:
            f.write(arcs_content)
        with open(coords_path, "w") as f:
            f.write(coords_content)

        # Appel de la fonction
        img = generate_plot_from_files(arcs_path, coords_path)

        # Tests
        assert isinstance(img, np.ndarray), "Image should be a NumPy array"
        assert img.shape[-1] == 3, "Image should be RGB"
