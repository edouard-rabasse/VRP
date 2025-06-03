import os
import tempfile
import torch
from omegaconf import OmegaConf
from src.graph.read_arcs import read_arcs
from src.graph.read_coordinates import read_coordinates
from src.graph.graph_flagging import flag_graph_from_instance


class DummyModel(torch.nn.Module):
    """A simple dummy model for testing — outputs two channels per image."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def create_fake_instance_files(tmpdir, number):
    arc_path = os.path.join(tmpdir, f"Arcs_{number}_1.txt")
    coord_path = os.path.join(tmpdir, f"Coordinates_{number}.txt")

    with open(arc_path, "w") as f:
        f.write("0;1;1;0\n1;2;2;0\n")

    with open(coord_path, "w") as f:
        f.write("0,1.0,1.0,10\n1,2.0,2.0,20\n2,3.0,1.0,0\n")

    return arc_path, coord_path


def test_flag_graph_from_instance(monkeypatch):
    number = 42

    with tempfile.TemporaryDirectory() as tmpdir:
        arc_path, coord_path = create_fake_instance_files(tmpdir, number)

        # Monkeypatch read_arcs and read_coordinates to read from our tmpdir
        monkeypatch.setattr(
            "src.graph.graph_flagging.read_arcs", lambda _: read_arcs(arc_path)
        )
        monkeypatch.setattr(
            "src.graph.graph_flagging.read_coordinates",
            lambda _: read_coordinates(coord_path),
        )

        # Dummy model
        model = DummyModel()
        model.eval()

        # Minimal config for heatmap
        cfg = OmegaConf.create(
            {
                "heatmap": {
                    "method": "gradcam",
                    "args": {
                        "class_index": 1,
                        "target_layer": "conv",  # matches DummyModel layer
                        "discard_ratio": 0.9,
                    },
                },
                "plot": {
                    "bounds": (-1, 11, -1, 11),  # Example bounds
                },
            }
        )

        flagged_arcs, flagged_coords = flag_graph_from_instance(
            instance_number=number,
            model=model,
            cfg=cfg,
            device="cpu",
        )

        # Assertions
        assert isinstance(flagged_arcs, list)
        assert all(len(a) == 5 for a in flagged_arcs)
        assert isinstance(flagged_coords, dict)
        assert all(len(v) == 4 for v in flagged_coords.values())
