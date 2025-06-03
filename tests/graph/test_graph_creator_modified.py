import os
import tempfile
import shutil
from PIL import Image

from src.graph.process_all import process_all_solutions


def test_process_all_solutions_creates_outputs(show=False):
    # Préparation des fichiers arcs et coordonnées factices
    arcs_content = "0;1;1;0;0\n1;2;2;0;1\n2;0;1;0;0\n"
    coords_content = "0,0.0,0.0,0.0,0\n1,1.0,0.0,0.0,0\n2,1.0,1.0,0.0,0\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        arcs_dir = os.path.join(tmpdir, "arcs")
        coords_dir = os.path.join(tmpdir, "coords")
        out_dir = os.path.join(tmpdir, "plots")

        os.makedirs(arcs_dir, exist_ok=True)
        os.makedirs(coords_dir, exist_ok=True)

        # Sauvegarde des fichiers
        arc_file = os.path.join(arcs_dir, "Arcs_1_0.txt")
        coord_file = os.path.join(coords_dir, "Coordinates_1.txt")
        with open(arc_file, "w") as f:
            f.write(arcs_content)
        with open(coord_file, "w") as f:
            f.write(coords_content)

        process_all_solutions(
            arcs_folder=arcs_dir,
            coordinates_folder=coords_dir,
            output_folder=out_dir,
            bounds=(-1, 11, -1, 11),
            valid_range=range(1, 2),
            type="modified",
            show_labels=False,
        )

        output_path = os.path.join(out_dir, "Plot_1.png")
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        if show:
            img = Image.open(output_path)
            img.show()
