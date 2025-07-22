import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from .iteration_result import IterationResult


class VRPInstance:
    """Represents a single VRP instance with its optimization results."""

    def __init__(self, instance_number: int, df: pd.DataFrame):
        self.instance_number = instance_number
        self.df = df
        self.iterations = self._parse_iterations()
        self.df = pd.DataFrame(self.iterations)

    def _parse_iterations(self) -> List[IterationResult]:
        """Parse DataFrame rows into IterationResult objects,
        adding previous entropy and classifier score info.
        """
        iterations = []
        first_time_valid = False

        for i, (_, row) in enumerate(self.df.iterrows()):
            row = row.copy()  # Important to avoid modifying the DataFrame

            if i > 0:
                row["entropy_previous"] = self.df.iloc[i - 1].get("entropy_score", None)
                row["classifier_score_previous"] = self.df.iloc[i - 1].get(
                    "classifier_score", None
                )
                row["top_arc_previous"] = self.df.iloc[i - 1].get("top_arc_value", None)
                row["top_3_arcs_previous"] = self.df.iloc[i - 1].get("top_3_arcs", None)
                if not first_time_valid and row["valid"]:
                    first_time_valid = True
                    row["first_time_valid"] = True
                else:
                    row["first_time_valid"] = False
            else:
                row["entropy_previous"] = None
                row["classifier_score_previous"] = None
                row["first_time_valid"] = False
                row["top_arc_previous"] = None
                row["top_3_arcs_previous"] = None
            iterations.append(IterationResult.from_pandas_series(row))

            if first_time_valid:
                break

        return iterations

    def get_best_iteration(self) -> IterationResult:
        """Get the iteration with the lowest score."""
        return min(self.iterations, key=lambda x: x.score)

    def get_valid_iterations(self) -> List[IterationResult]:
        """Get all valid iterations."""
        return [iter_result for iter_result in self.iterations if iter_result.valid]

    def get_first_valid_iteration(self) -> Optional[IterationResult]:
        """Get the first valid iteration."""
        for iter_result in self.iterations:
            if iter_result.valid:
                return iter_result
        return None

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for this instance, focusing on meaningful metrics."""

        valid_iterations = self.get_valid_iterations()
        first_valid = self.get_first_valid_iteration()
        final_iteration = self.iterations[-1]

        stats = {
            "instance_number": self.instance_number,
            "total_iterations": len(self.iterations),
            "total_time": sum(it.time for it in self.iterations if it.time is not None),
            # Validation metrics
            "first_valid_at_iter": first_valid.iter if first_valid else None,
            # Cost progression
            "first_valid_costs": {
                "config7_cost": first_valid.config7_cost if first_valid else None,
                "solver_cost": first_valid.solver_cost if first_valid else None,
                "easy_cost": first_valid.easy_cost if first_valid else None,
            },
            # Key iteration snapshots
            "snapshots": {
                "initial": self._get_iteration_snapshot(self.iterations[0]),
                "final": self._get_iteration_snapshot(final_iteration),
            },
        }

        # Add first valid iteration snapshot if it exists
        if first_valid:
            stats["snapshots"]["first_valid"] = self._get_iteration_snapshot(
                first_valid
            )

        # Violation analysis
        violations = [
            it.number_of_violations
            for it in self.iterations
            if it.number_of_violations is not None
        ]
        if violations:
            stats["violations"] = {
                "initial": violations[0] if violations else None,
                "final": violations[-1] if violations else None,
                "max_violations": max(violations),
                "iterations_with_violations": sum(1 for v in violations if v > 0),
            }

        return stats

    def _get_iteration_snapshot(self, iteration: IterationResult) -> Dict:
        """Get a snapshot of key metrics for a specific iteration."""
        snapshot = {
            "iter": iteration.iter,
            "time": iteration.time,
            "valid": iteration.valid,
            "solver_cost": iteration.solver_cost,
            "number_of_violations": iteration.number_of_violations,
        }

        # Add ML metrics if available
        if iteration.classifier_score is not None:
            snapshot["classifier_score"] = iteration.classifier_score
        if iteration.entropy_score is not None:
            snapshot["entropy_score"] = iteration.entropy_score
        if iteration.top_arc_value is not None:
            snapshot["top_arc_value"] = iteration.top_arc_value
        if iteration.top_3_arcs is not None:
            snapshot["top_3_arcs"] = iteration.top_3_arcs
        if iteration.entropy_previous is not None:
            snapshot["entropy_previous"] = iteration.entropy_previous
        if iteration.classifier_score_previous is not None:
            snapshot["classifier_score_previous"] = iteration.classifier_score_previous
        if iteration.entropy_variation is not None:
            snapshot["entropy_variation"] = iteration.entropy_variation
        if iteration.classifier_score_variation is not None:
            snapshot["classifier_score_variation"] = (
                iteration.classifier_score_variation
            )
        if iteration.top_arc_variation is not None:
            snapshot["top_arc_variation"] = iteration.top_arc_variation
        if iteration.top_3_arcs_variation is not None:
            snapshot["top_3_arcs_variation"] = iteration.top_3_arcs_variation

        return snapshot

    def get_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame representation of this instance's iterations."""
        return pd.DataFrame([it.to_dict() for it in self.iterations])
