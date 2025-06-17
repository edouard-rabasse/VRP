# File: src/pipeline/optimized_pipeline.py

## TODO: Create a copy before launching in the right folder
## TODO: retrieve costs with java and all


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
        self.files = FileService(
            BASE_DIR, self.cfg.solver.instance_folder, self.cfg.solver.result_folder
        )
        self.solver = SolverClient(
            msh_dir=BASE_DIR,
            java_lib=Path(self.cfg.solver.java_lib),
            program_name=self.cfg.solver.program_name,
            custom_args=self.cfg.solver.custom_args,
        )

    def flag_arcs(
        self, instance: int, suffix: str, config_number="1"
    ) -> tuple[list, list]:
        coords, depot = self.files.load_coordinates(instance)
        arcs = self.files.load_arcs(
            instance, config_number=config_number, suffix=suffix
        )
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
        self.model.eval()
        with torch.no_grad():
            out = self.model(tensor)
            score = torch.sigmoid(out).squeeze().cpu()[1].item()
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
        thresh: float = 0.5,
    ) -> dict:
        results = {"best_score": 1, "iterations": []}
        start = current_time()
        iteration = 1
        score = 0
        results["converged"] = False

        # Create copy of the arc file
        self.files.create_copy(
            instance,
            org_cfg_number=self.cfg.solver.org_config,
            cfg_number=self.cfg.solver.config,
        )

        while iteration < max_iter + 1:
            print(f"[Debug] Iteration {iteration} for instance {instance}")
            t0 = current_time()

            flagged_arcs, flagged_coords = self.flag_arcs(
                instance, suffix=iteration, config_number=self.cfg.solver.config
            )

            self.files.save_arcs(
                instance,
                flagged_arcs,
                config_number=self.cfg.solver.config,
                suffix=iteration,
            )
            coords, depot = self.files.load_coordinates(instance)
            arcs = self.files.load_arcs(
                instance, config_number=self.cfg.solver.config, suffix=iteration
            )
            # score = self.score(flagged_coords, flagged_arcs, depot)
            score = self.score(coords, arcs, depot)
            dt = current_time() - t0
            results["iterations"].append(
                {"iter": iteration, "score": score, "time": dt}
            )

            if score < results["best_score"]:
                improvement = -(results["best_score"] - score) / results["best_score"]
                results["best_score"] = score
                # plot_routes(
                #     flagged_arcs,
                #     flagged_coords,
                #     depot,
                #     output_file=f"{self.cfg.solver.plot_output_folder}/instance_{instance}_iter_{iteration}.png",
                #     bounds=tuple(self.cfg.plot.bounds),
                #     route_type="modified",
                #     show_labels=True,
                # )
            if iteration > 1 and score < thresh:
                results["converged"] = True
                results["number_iter"] = iteration

                break

            self.run_vrp_solver(
                instance, arc_suffix=iteration, config_name=self.cfg.solver.config_name
            )
            iteration += 1

        results = self.check_final(results, instance, iteration)
        results["total_time"] = current_time() - start
        # self.files.delete_intermediate_files(
        #     instance, config_number=self.cfg.solver.config, iter=iteration
        # )
        return results

    def check_final(self, results: dict, instance: int, iteration: int) -> None:
        """Check final results for convergence and cost analysis.

        Args:
            results (dict): Dictionary to store results.
            instance (int): Instance number.
            iteration (int): Current iteration number.

        Returns:
            dict: Updated results dictionary with convergence status and cost analysis.
        """
        results["converged"] = True
        cost_analysis = self.files.load_results(
            instance, iteration, self.cfg.solver.config
        )

        arcs_init = self.files.load_arcs(
            instance, config_number=self.cfg.solver.org_config, suffix="1"
        )
        arcs_final = self.files.load_arcs(
            instance,
            config_number=self.cfg.solver.config,
            suffix=iteration,  # the last index
        )
        results["initial_costs"] = cost_analysis["OldCost"]
        results["final_costs"] = cost_analysis["NewCost"]
        results["cost_delta"] = (
            results["final_costs"] - results["initial_costs"]
        ) / results["initial_costs"]
        results["valid"] = cost_analysis["Valid"]
        results["equal"] = self.files.compare_arcs(arcs_init, arcs_final)
        return results
