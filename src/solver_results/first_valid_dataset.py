import pandas as pd
from typing import List, Optional
from .vrp_instance import VRPInstance


class FirstValidDataset:
    """Dataset focused on first valid iteration per instance."""

    def __init__(self, vrp_instances: List[VRPInstance]):
        self.vrp_instances = vrp_instances
        self.dataset_df = self._create_dataset()
        self.compute_cost_difference()

    def _create_dataset(self) -> pd.DataFrame:
        """Create dataset with one row per instance containing first valid metrics."""
        data_rows = []

        for instance in self.vrp_instances:
            first_valid = instance.get_first_valid_iteration()

            if first_valid is not None:
                # Instance has a first valid iteration
                row_data = {
                    "instance_number": instance.instance_number,
                    "has_first_valid": True,
                    "first_valid_iter": first_valid.iter,
                    "first_valid_time": first_valid.time,
                    **first_valid.to_dict(),  # Include all metrics from first valid iteration
                }

            else:
                # Instance has no valid iterations
                row_data = {
                    "instance_number": instance.instance_number,
                    "has_first_valid": False,
                    "first_valid_iter": None,
                    "first_valid_time": None,
                    # Set all other metrics to None
                    "valid": None,
                    "config7_cost": None,
                    "solver_cost": None,
                    "easy_cost": None,
                    "number_of_violations": None,
                    "classifier_score": None,
                    "entropy_score": None,
                    "top_arc_value": None,
                    "top_3_arcs": None,
                    "entropy_variation": None,
                    "classifier_score_variation": None,
                    "top_arc_variation": None,
                    "top_3_arcs_variation": None,
                }

            data_rows.append(row_data)

        return pd.DataFrame(data_rows)

    def get_dataset_summary(self) -> dict:
        """Get summary statistics of the dataset."""
        total_instances = len(self.vrp_instances)
        instances_with_valid = self.dataset_df["has_first_valid"].sum()
        instances_without_valid = total_instances - instances_with_valid

        return {
            "total_instances": total_instances,
            "instances_with_first_valid": instances_with_valid,
            "instances_without_first_valid": instances_without_valid,
            "percentage_with_valid": (
                (instances_with_valid / total_instances * 100)
                if total_instances > 0
                else 0
            ),
        }

    def get_valid_instances_df(self) -> pd.DataFrame:
        """Get DataFrame containing only instances with first valid iterations."""
        return self.dataset_df[self.dataset_df["has_first_valid"] == True].copy()

    def get_invalid_instances_df(self) -> pd.DataFrame:
        """Get DataFrame containing only instances without valid iterations."""
        return self.dataset_df[self.dataset_df["has_first_valid"] == False].copy()

    def get_full_dataset(self) -> pd.DataFrame:
        """Get the complete dataset."""
        return self.dataset_df.copy()

    def print_summary(self):
        """Print a detailed summary of the dataset."""
        summary = self.get_dataset_summary()

        print("=" * 50)
        print("FIRST VALID DATASET SUMMARY")
        print("=" * 50)
        print(f"Total VRP instances: {summary['total_instances']}")
        print(f"Instances with first valid: {summary['instances_with_first_valid']}")
        print(f"Instances without valid: {summary['instances_without_first_valid']}")
        print(f"Success rate: {summary['percentage_with_valid']:.1f}%")

        if summary["instances_with_first_valid"] > 0:
            valid_df = self.get_valid_instances_df()
            print(f"\nFirst Valid Iteration Statistics:")
            print(
                f"  Average iteration to first valid: {valid_df['first_valid_iter'].mean():.1f}"
            )
            print(
                f"  Min iteration to first valid: {valid_df['first_valid_iter'].min()}"
            )
            print(
                f"  Max iteration to first valid: {valid_df['first_valid_iter'].max()}"
            )
            # print(
            #     f"  Average time to first valid: {valid_df['first_valid_time'].mean():.2f}s"
            # )

            # Cost statistics
            if "solver_cost" in valid_df.columns:
                print(f"\nCost Statistics (first valid):")
                print(
                    f"  Average solver cost difference : {valid_df['difference_with_solver'].mean():.2f}"
                )
                print(
                    f"  Average easy cost difference: {valid_df['difference_with_easy'].mean():.2f}"
                )

    def save_to_csv(self, filepath: str):
        """Save the dataset to CSV."""
        self.dataset_df.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")

    def compute_cost_difference(self):
        """Compute the difference between solver cost and config7 cost."""
        try:
            if (
                "solver_cost" in self.dataset_df.columns
                and "config7_cost" in self.dataset_df.columns
            ):
                self.dataset_df["difference_with_solver"] = (
                    self.dataset_df["solver_cost"] - self.dataset_df["config7_cost"]
                )
                self.dataset_df["difference_with_easy"] = (
                    self.dataset_df["easy_cost"] - self.dataset_df["solver_cost"]
                )
                print("Cost difference computed and added to dataset.")
        except Exception as e:
            print(f"Error computing cost difference: {e}")
            print("Solver cost or config7 cost not found in dataset.")
