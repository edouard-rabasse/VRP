import os
import tempfile
import numpy as np
import torch
import pytest
from types import SimpleNamespace
from PIL import Image

import src.pipeline.optimized_pipeline as op
from src.pipeline.optimized_pipeline import OptimizedVRPPipeline

# -- Fixtures & helpers ---------------------------------------------------


@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch, tmp_path):
    # Stub get_cfg to return a minimal config
    fake_cfg = SimpleNamespace(
        model=SimpleNamespace(name="dummy", **{}),
        plot=SimpleNamespace(bounds=[0, 5, 0, 5]),
        data=SimpleNamespace(),
        batch_size=1,
        model_params=SimpleNamespace(epochs=1, learning_rate=0.1),
        save_model=False,
        experiment_name="exp",
        data_cfg_number=1,
        solver=SimpleNamespace(
            java_lib="dummy_solver.jar",
            config_name="conf.xml",
            max_iterations=10,
            threshold=0.1,
            config="1",
            program_name="main.Main_customCosts",
            custom_args=["-Xmx14000m", "-Djava.library.path=dummy_solver.jar"],
        ),
    )
    monkeypatch.setattr(op, "get_cfg", lambda overrides=None: fake_cfg)

    # Stub ModelLoader.load
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # return 2-class logits: batch x 2
            return torch.tensor([[0.1, 0.9]])

    monkeypatch.setattr(
        op,
        "ModelLoader",
        lambda cfg, device: SimpleNamespace(load=lambda: DummyModel()),
    )

    # Stub FileService
    fs = SimpleNamespace(
        load_coordinates=lambda ins: ([(0.0, 0.0), (1.0, 1.0)], 0),
        load_arcs=lambda ins, config_number, suffix: [(0, 1, 1, 0)],
        save_arcs=lambda *args, **kw: None,
    )
    monkeypatch.setattr(op, "FileService", lambda base: fs)

    # Stub SolverClient
    sc = SimpleNamespace(run=lambda *args, **kw: None)
    monkeypatch.setattr(op, "SolverClient", lambda *args, **kwargs: sc)

    # Stub graph and plotting
    monkeypatch.setattr(
        op,
        "flag_graph_from_data",
        lambda arcs, coords, depot, model, cfg, device: (arcs, coords),
    )
    monkeypatch.setattr(
        op,
        "generate_plot_from_dict",
        lambda arcs, coordinates, depot, bounds: np.zeros((5, 5, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(op, "plot_routes", lambda *args, **kw: None)

    # Stub image_transform_test
    monkeypatch.setattr(
        op, "image_transform_test", lambda: lambda img: torch.randn(1, 3, 5, 5)
    )

    # Ensure a clean output directory
    monkeypatch.setattr(op, "BASE_DIR", str(tmp_path))


# -- Tests ---------------------------------------------------


def test_flag_arcs_returns_flagged(monkeypatch):
    pipeline = OptimizedVRPPipeline()
    flagged, flagged_coords = pipeline.flag_arcs(
        instance=123, suffix="1", config_number="1"
    )
    # we stubbed flag_graph_from_data to return input arcs and coords
    assert flagged == [(0, 1, 1, 0)]
    assert flagged_coords == [(0.0, 0.0), (1.0, 1.0)]


def test_score_computes_sigmoid(monkeypatch):
    pipeline = OptimizedVRPPipeline()
    # use dummy coords/arcs
    coords = [(0, 0), (1, 1)]
    arcs = [(0, 1, 1, 0)]
    score = pipeline.score(coords, arcs, depot=0)
    # our dummy model returns logits [0.1,0.9], sigmoid of 0.9 ~ 0.7109
    assert pytest.approx(torch.sigmoid(torch.tensor(0.9)).item(), rel=1e-3) == score


def test_run_vrp_solver_invokes_solver(monkeypatch):
    pipeline = OptimizedVRPPipeline()
    called = {}

    def fake_run(instance, config_name, arc_suffix):
        called["args"] = (instance, config_name, arc_suffix)

    pipeline.solver = SimpleNamespace(run=fake_run)
    pipeline.run_vrp_solver(42, config_name="conf.xml", arc_suffix="2")
    assert called["args"] == (42, "conf.xml", "2")


def test_iterative_optimization_stops_on_threshold():
    pipeline = OptimizedVRPPipeline()
    # stub flag_arcs and score
    pipeline.flag_arcs = lambda *args, **kwargs: ([(0, 1, 1, 0)], [(0, 0), (1, 1)])
    # first iteration returns 0.5, second returns 0.1
    scores = [0.5, 0.1]
    pipeline.score = lambda *args, **kwargs: scores.pop(0)
    called = []
    pipeline.run_vrp_solver = lambda inst, config_name, arc_suffix=None: called.append(
        arc_suffix
    )

    results = pipeline.iterative_optimization(instance=7, max_iter=3, thresh=0.2)

    assert results["best_score"] == pytest.approx(0.1)
    # two iterations logged
    assert [it["score"] for it in results["iterations"]] == [0.5, 0.1]
    assert results.get("converged", False) is True
    # suffixes recorded as integers
    assert called == [1]


def test_iterative_optimization_defaults_runs_full(monkeypatch):
    pipeline = OptimizedVRPPipeline()
    pipeline.flag_arcs = lambda *args, **kwargs: ([(0, 1, 1, 0)], [(0, 0), (1, 1)])
    pipeline.score = lambda *args, **kwargs: 1.0
    pipeline.run_vrp_solver = lambda *args, **kwargs: None
    # no-op for plotting
    op.plot_routes = lambda *args, **kwargs: None

    results = pipeline.iterative_optimization(instance=5, max_iter=2, thresh=0.0)

    # Should have two iterations since no early stop
    assert len(results["iterations"]) == 2
    assert results["best_score"] == 1.0
    assert "converged" not in results
