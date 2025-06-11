# File: src/pipeline/optimized_pipeline.py


import time
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.pipeline.config import get_cfg, BASE_DIR
from src.pipeline.model_loader import ModelLoader
from src.pipeline.file_service import FileService
from src.pipeline.solver_client import SolverClient
from src.transform import image_transform_test
from src.graph.graph_flagging import flag_graph_from_data
from src.graph import generate_plot_from_dict, plot_routes


def current_time():
    return time.perf_counter()


class OptimizedVRPPipeline:
    def __init__(self, config_overrides: list[str] | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = get_cfg(config_overrides)

        # Services
        self.model = ModelLoader(self.cfg.model, self.device).load()
        self.files = FileService(BASE_DIR)
        self.solver = SolverClient(
            msh_dir=BASE_DIR,
            java_lib=Path("C:/gurobi1201/win64/bin"),
        )

    def flag_arcs(self, instance: int, suffix: str, config="1") -> tuple[list, list]:
        coords, depot = self.files.read_coordinates(instance)
        arcs = self.files.read_arcs(instance, config=config, suffix=suffix)
        flagged, flagged_coords = flag_graph_from_data(
            arcs, coords, depot, self.model, self.cfg, device=self.device
        )
        return flagged, flagged_coords

    def score(self, coords: list[tuple[float, float]], arcs: list, depot) -> float:
        img = generate_plot_from_dict(
            arcs, coordinates=coords, depot=depot, bounds=tuple(self.cfg.plot.bounds)
        )

        tensor = (
            image_transform_test()(Image.fromarray(img)).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            out = self.model(tensor)
            score = torch.sigmoid(out).squeeze().cpu()[1].item()
        print(f"[Debug] Scoring for instance: {score}")
        return score

    def run_vrp_solver(
        self,
        instance: int,
        config_name: str = "configurationCustomCosts2.xml",
        arc_suffix: str = "1",
    ) -> None:
        self.solver.run(
            instance,
            config_name=config_name,
            arc_suffix=arc_suffix,
        )

    def iterative_optimization(
        self,
        instance: int,
        max_iter: int = 5,
        thresh: float = 0.2,
    ) -> dict:
        results = {"best_score": 1, "iterations": []}
        start = current_time()
        iteration = 1
        score = 0
        while iteration < max_iter + 1:
            print(f"[Debug] Iteration {iteration} for instance {instance}")
            t0 = current_time()

            flagged_arcs, flagged_coords = self.flag_arcs(
                instance, suffix=iteration, config="CustomCosts"
            )

            self.files.save_arcs(
                instance, flagged_arcs, config="CustomCosts", suffix=iteration
            )
            coords, depot = self.files.read_coordinates(instance)
            arcs = self.files.read_arcs(
                instance, config="CustomCosts", suffix=iteration
            )
            score = self.score(flagged_coords, flagged_arcs, depot)
            dt = current_time() - t0
            results["iterations"].append(
                {"iter": iteration, "score": score, "time": dt}
            )
            plot_routes(
                flagged_arcs,
                flagged_coords,
                depot,
                output_file=f"output/test/instance_{instance}_iter_{iteration}.png",
                bounds=tuple(self.cfg.plot.bounds),
                route_type="modified",
                show_labels=True,
            )
            if score < results["best_score"]:
                improvement = -(results["best_score"] - score) / results["best_score"]
                results["best_score"] = score
                if iteration > 1 and score < thresh:
                    results["converged"] = True
                    break

            self.run_vrp_solver(instance, arc_suffix=iteration)
            iteration += 1
        results["total_time"] = current_time() - start
        return results
