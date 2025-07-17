# File: src/pipeline/optimized_pipeline.py

## TODO: Create a copy before launching in the right folder
## TODO: retrieve costs with java and all


import time
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import numpy as np

from src.pipeline.config import BASE_DIR, override_java_param
from src.pipeline.model_loader import ModelLoader
from src.pipeline.file_service import FileService
from src.pipeline.solver_client import SolverClient
from src.transform import image_transform_test
from src.graph.graph_flagging import flag_graph_from_data
from src.graph import generate_plot_from_dict, plot_routes

from src.transform import image_transform_test
from src.visualization import get_heatmap


def current_time():
    return time.perf_counter()


class OptimizedVRPPipeline:
    def __init__(self, cfg: DictConfig | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg

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
        self,
        instance: int,
        suffix: str | int,
        config_number="1",
        return_weighted_sum=False,
        top_n_arcs=1,
        threshold: float = 0.2,
        heatmap: np.ndarray | None = None,
    ) -> tuple[list, list, np.ndarray]:
        coords, depot = self.files.load_coordinates(instance)
        arcs = self.files.load_arcs(
            instance, config_number=config_number, suffix=suffix
        )

        image = generate_plot_from_dict(
            arcs, coords, depot=depot, bounds=tuple(self.cfg.plot.bounds)
        )
        input_tensor = (
            image_transform_test()(Image.fromarray(image)).unsqueeze(0).to(self.device)
        )

        heatmap = get_heatmap(
            model=self.model,
            method=self.cfg.heatmap.method,
            input_tensor=input_tensor,
            args=self.cfg.heatmap.args,
            device=self.device,
        )
        flagged, flagged_coords = flag_graph_from_data(
            arcs,
            coords,
            depot,
            self.model,
            self.cfg,
            device=self.device,
            return_weighted_sum=return_weighted_sum,
            top_n_arcs=top_n_arcs,
            threshold=threshold,
            heatmap=heatmap,
        )
        return flagged, flagged_coords, heatmap

    def score(self, coords: dict, arcs: list, depot) -> float:
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
        arc_suffix: str | int = "1",
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

            flagged_arcs, flagged_coords, heatmap = self.flag_arcs(
                instance,
                suffix=iteration,
                config_number=self.cfg.solver.config,
                return_weighted_sum=self.cfg.solver.return_weighted_sum,
                top_n_arcs=self.cfg.solver.top_n_arcs,
                threshold=self.cfg.solver.heatmap_threshold,
            )
            coords, depot = self.files.load_coordinates(instance)
            arcs = self.files.load_arcs(
                instance, config_number=self.cfg.solver.config, suffix=iteration
            )

            best_arc_value = self.extract_top_arc_value(flagged_arcs, index=4)

            flagged_arcs = self.files.binarize_arcs(
                flagged_arcs,
                threshold=0,
                index=4,
            )

            self.files.save_arcs(
                instance,
                flagged_arcs,
                config_number=self.cfg.solver.config,
                suffix=iteration,
            )

            # score = self.score(flagged_coords, flagged_arcs, depot)
            score = self.score(coords, arcs, depot)

            heatmap_entropy = self._score_heatmap(heatmap)

            dt = current_time() - t0

            if score < results["best_score"]:
                results["best_score"] = score

            self._generate_plot_if_needed(
                flagged_arcs, flagged_coords, depot, instance, iteration
            )

            cost_analysis = None
            if iteration > 1:
                cost_analysis = self.files.load_results(
                    instance, iteration, self.cfg.solver.config
                )
            valid = cost_analysis["Valid"] if cost_analysis else False
            if self._check_convergence(score, thresh, valid, results):
                results["converged"] = True
                results["number_iter"] = iteration

            results["iterations"].append(
                self._create_iteration_result(
                    iteration,
                    score,
                    dt,
                    best_arc_value,
                    cost_analysis,
                    score_heatmap=heatmap_entropy,
                )
            )

            # Run slver for next iteration
            self.run_vrp_solver(
                instance, arc_suffix=iteration, config_name=self.cfg.solver.config_name
            )
            iteration += 1

        results = self.check_final(results, instance, iteration)
        results["total_time"] = current_time() - start

        # Cleanup
        self.files.delete_intermediate_files(
            instance, config_number=self.cfg.solver.config, iter=iteration
        )
        return results

    def _generate_plot_if_needed(
        self,
        flagged_arcs: list,
        flagged_coords: dict,
        depot,
        instance: int,
        iteration: int,
    ) -> None:
        """Generate plot if plotting is enabled."""
        if self.cfg.solver.plot:
            plot_routes(
                flagged_arcs,
                flagged_coords,
                depot,
                output_file=f"{self.cfg.solver.plot_output_folder}/instance_{instance}_iter_{iteration}.png",
                bounds=tuple(self.cfg.plot.bounds),
                route_type="modified",
                show_labels=True,
            )

    def _create_iteration_result(
        self,
        iteration: int,
        score: float,
        dt: float,
        best_arc_value: float,
        cost_analysis: dict = None,
        score_heatmap: float = 0.0,
    ) -> dict:
        """Create a standardized iteration result dictionary."""
        if cost_analysis and iteration > 1:
            return {
                "iter": iteration,
                "score": score,
                "time": dt,
                "best_arc_value": best_arc_value,
                "valid": cost_analysis["Valid"],
                "config7_cost": cost_analysis["OldCost"],
                "solver_cost": cost_analysis["NewCost"],
                "easy_cost": cost_analysis["EasyCost"],
                "number_of_violations": cost_analysis["numberOfViolations"],
                "score_heatmap": score_heatmap,
            }
        else:
            return {
                "iter": iteration,
                "score": score,
                "time": dt,
                "best_arc_value": best_arc_value,
                "valid": False,
                "config7_cost": 0,
                "solver_cost": 0,
                "easy_cost": 0,
                "number_of_violations": None,
                "score_heatmap": score_heatmap,
            }

    def _check_convergence(
        self, score: float, thresh: float, valid: bool, results: dict
    ) -> bool:
        """Check if optimization should converge."""
        return score < thresh or (valid and not results["converged"])

    def check_final(self, results: dict, instance: int, iteration: int) -> dict:
        """Check final results for convergence and cost analysis.

        Args:
            results (dict): Dictionary to store results.
            instance (int): Instance number.
            iteration (int): Current iteration number.

        Returns:
            dict: Updated results dictionary with convergence status and cost analysis.
        """
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
        results["easy_costs"] = cost_analysis["EasyCost"]
        results["cost_delta"] = (
            results["final_costs"] - results["initial_costs"]
        ) / results["initial_costs"]
        results["delta_easy"] = (
            results["easy_costs"] - results["final_costs"]
        ) / results["final_costs"]
        results["valid"] = cost_analysis["Valid"]
        results["equal"] = self.files.compare_arcs(arcs_init, arcs_final)
        return results

    def _score_heatmap(
        self,
        heatmap: np.ndarray,
    ) -> float:
        """Compute the entropy of the heatmap as a score."""
        if heatmap is None:
            return 0.0
        # Normalize the heatmap to avoid division by zero
        heatmap = heatmap / np.sum(heatmap) if np.sum(heatmap) > 0 else heatmap
        # Compute entropy as a score
        score = -np.sum(heatmap * np.log(heatmap + 1e-10))
        return score

    def run_optimized_pipeline(
        self,
        walking: int,
        multiplier: float,
        threshold: float,
        numbers: list,
        max_iter: int = 100,
        output_dir: str = "output",
    ):
        """
        Run the optimized VRP pipeline with specified parameters.
        Args:
            walking (int): The walking cost to set in the Java configuration.
            multiplier (float): The multiplier for the custom costs.
            threshold (float): The convergence threshold for the optimization.
            numbers (list): List of instance numbers to process.
        Returns:
            None

        """
        print(
            f"Running with walking cost: {walking}, multiplier: {multiplier}, threshold: {threshold}"
        )
        override_java_param(
            "MSH/MSH/config/configurationCustomCosts2.xml",
            {
                "DEFAULT_WALK_COST": walking,
                "CUSTOM_COST_MULTIPLIER": multiplier,
            },
        )
        nb_equal = 0
        nb_total = 0
        nb_valid = 0
        average_diff = 0.0
        easy_diff = 0.0
        nb_converged = 0
        nb_iter = 0

        start = current_time()
        for i in numbers:

            res = self.iterative_optimization(
                instance=i, max_iter=max_iter, thresh=threshold
            )
            nb_equal += res["equal"]
            nb_total += 1
            nb_valid += res["valid"]
            if res["valid"]:
                average_diff += res["cost_delta"]
                easy_diff += res["delta_easy"]
            if res["converged"]:
                nb_converged += 1
                nb_iter += res["number_iter"]

            (output_name,) = (
                f"{output_dir}/{self.cfg.model.name}_{walking}_{multiplier}_{threshold}/instance_{i}.csv",
            )

            self.files.write_list_to_csv(output_name, res["iterations"])

        end = current_time()
        total_time = end - start

        # Save results in a file
        self.write_results(
            nb_total=nb_total,
            nb_valid=nb_valid,
            nb_equal=nb_equal,
            average_diff=average_diff,
            easy_diff=easy_diff,
            output_dir=output_dir,
            walking=walking,
            multiplier=multiplier,
            threshold=threshold,
            nb_converged=nb_converged,
            nb_iter=nb_iter,
            total_time=total_time,
        )

    def write_results(
        self,
        *,
        nb_total: int,
        nb_valid: int,
        nb_equal: int,
        average_diff: float,
        easy_diff: float,
        output_dir: str,
        walking: int,
        multiplier: float,
        threshold: float,
        nb_converged: int,
        nb_iter: int,
        total_time: float,
    ):
        nb_diff = nb_total - nb_valid
        average_diff /= nb_valid if nb_valid > 0 else 0.0

        # save results in a file
        with open(
            f"{output_dir}/results_{walking}_{multiplier}_{threshold}.txt", "w"
        ) as f:
            f.write(f"Walking cost: {walking}\n")
            f.write(f"Multiplier: {multiplier}\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Total instances: {nb_total}\n")
            f.write(f"Valid instances: {nb_valid}\n")
            f.write(f"Equal instances: {nb_equal}\n")
            f.write(f"Average cost difference with perfect optim: {average_diff:.2f}\n")
            f.write(f"Average cost difference with easy optim: {easy_diff:.2f}\n")

            f.write(f"Converged instances: {nb_converged}\n")
            f.write(
                f"Average iterations for converged instances: {nb_iter / nb_converged if nb_converged > 0 else 0:.2f}\n"
            )
            f.write(f"Total time taken: {total_time:.2f} seconds\n")

    def extract_top_arc_value(self, flagged_arcs: list, index: int = 4) -> list:
        """
        Extract the top N arcs from the flagged arcs based on their weights.
        Args:
            flagged_arcs (list): List of flagged arcs with weights.
            top_n (int): Number of top arcs to extract.
        Returns:
            list: List of top N arcs.
        """
        sorted_arcs = sorted(flagged_arcs, key=lambda x: x[index], reverse=True)
        return sorted_arcs[0][index]
