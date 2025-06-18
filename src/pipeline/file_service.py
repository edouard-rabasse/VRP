from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.graph import read_coordinates, read_arcs, load_set_arc


class FileService:
    """
    Centralized file path management and I/O operations for VRP instances and solutions.

    Handles file paths, reading coordinates and arcs, and saving results for VRP instances
    """

    ""

    def __init__(
        self,
        base_dir: Path,
        instance_folder: str,
        results_folder: str,
    ):
        self.base_dir = Path(base_dir)
        self.instance_folder = instance_folder
        self.results_folder = results_folder

    def get_coordinates_path(self, instance: int) -> Path:
        """
        Get the path to the coordinates file for a given instance.

        Args:
            instance: Instance number

        Returns:
            Path to the coordinates file

        Raises:
            FileNotFoundError: If the coordinates file doesn't exist
        """
        coordinates_path = (
            self.base_dir / self.instance_folder / f"Coordinates_{instance}.txt"
        )

        if not coordinates_path.exists():
            raise FileNotFoundError(f"Coordinates file not found: {coordinates_path}")

        return coordinates_path

    def get_arcs_path(
        self,
        instance: int,
        config_number: str,
        suffix: str,
    ) -> Path:
        """
        Get the path to the arcs file for a given instance and configuration.

        Args:
            instance: Instance number
            config: Configuration identifier
            suffix: File suffix

        Returns:
            Path to the arcs file
        """
        return (
            self.base_dir
            / self.results_folder
            / f"configuration{config_number}"
            / f"Arcs_{instance}_{suffix}.txt"
        )

    def get_results_path(
        self,
        instance: int,
        config_number: str,
        suffix: str,
    ) -> Path:
        """
        Get the path to the results file for a given instance and configuration.

        Args:
            instance: Instance number
            config: Configuration identifier
            suffix: File suffix

        Returns:
            Path to the results file
        """
        return (
            self.base_dir
            / self.results_folder
            / f"configuration{config_number}"
            / f"CostAnalysis_{instance}_{suffix}.txt"
        )

    # def get_solution_arcs_path(
    #     self,
    #     instance: int,
    #     suffix: str = DEFAULT_SUFFIX,
    #     config_number: str = CUSTOM_COSTS_CONFIG,
    # ) -> Path:
    #     """
    #     Get the path to the solution arcs file.

    #     Args:
    #         instance: Instance number
    #         suffix: File suffix
    #         config: Configuration identifier

    #     Returns:
    #         Path to the solution arcs file
    #     """
    #     return self.get_arcs_path(instance, config_number, suffix)

    def load_coordinates(
        self,
        instance: int,
        coordinate_type: str = "original",
        keep_service_time: bool = False,
    ) -> Tuple[dict, int]:
        """
        Load coordinates from file for a given instance.

        Args:
            instance: Instance number
            coordinate_type: Type of coordinates to load
            keep_service_time: Whether to keep service time information

        Returns:
            Tuple of (coordinates dictionary, depot count)
        """
        coordinates_path = self.get_coordinates_path(instance)
        return read_coordinates(
            str(coordinates_path), coordinate_type, keep_service_time
        )

    def get_cost_path(self, instance: int, config_number: str, suffix: str) -> Path:
        """
        Get the path to the costs file for a given instance.

        Args:
            instance: Instance number
            suffix: File suffix

        Returns:
            Path to the costs file
        """
        return (
            self.base_dir
            / self.results_folder
            / f"configuration{config_number}"
            / f"Costs_{instance}_{suffix}.txt"
        )

    def read_costs(self, path) -> Dict[int, Dict[int, float]]:
        """
        Read costs from a file and return as a dictionary.

        Args:
            path: Path to the costs file, containing

        Returns:
            Dictionary of costs
        """
        costs = {}

    def load_arcs(
        self,
        instance: int,
        config_number: str,
        suffix: str,
        arc_type: str = "original",
    ) -> List[Tuple]:
        """
        Load arcs from file for a given instance and configuration.

        Args:
            instance: Instance number
            config: Configuration identifier
            suffix: File suffix
            arc_type: Type of arcs to load

        Returns:
            List of arc tuples
        """
        arcs_path = self.get_arcs_path(instance, config_number, suffix)
        return read_arcs(str(arcs_path), arc_type)

    def save_arcs(
        self,
        instance: int,
        arcs: List[Tuple],
        config_number: str,
        suffix: str,
    ) -> None:
        """
        Save arcs to file for a given instance and configuration.

        Args:
            instance: Instance number
            arcs: List of arc tuples to save
            config: Configuration identifier
            suffix: File suffix
        """
        arcs_path = self.get_arcs_path(instance, config_number, suffix)
        self._ensure_directory_exists(arcs_path.parent)

        with open(arcs_path, "w", encoding="utf-8") as file:
            for arc in arcs:
                self._write_arc_line(file, arc)

    def _ensure_directory_exists(self, directory: Path) -> None:
        """Ensure the directory exists, creating it if necessary."""
        directory.mkdir(parents=True, exist_ok=True)

    def _write_arc_line(self, file, arc: Tuple) -> None:
        """Write a single arc line to the file."""
        arc_data = ";".join(str(value) for value in arc)
        file.write(f"{arc_data}\n")

    def create_copy(
        self,
        instance,
        org_cfg_number: str,
        cfg_number: str,
        suffix: str = "1",
    ) -> None:
        """Creates a copy of the results file for a given instance."""
        original_path = self.get_arcs_path(instance, org_cfg_number, suffix)
        copy_path = self.get_arcs_path(instance, cfg_number, suffix)

        if not original_path.exists():
            raise FileNotFoundError(f"Original file not found: {original_path}")

        copy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(original_path, "r", encoding="utf-8") as original_file:
            with open(copy_path, "w", encoding="utf-8") as copy_file:
                copy_file.write(original_file.read())

    def load_results(self, instance, iteration, config_number):
        """
        Load results for a given instance and iteration.

        Args:
            instance: Instance number
            iteration: Iteration number

        Returns:
            Tuple of (arcs, coordinates, depot)
        """
        path = self.get_results_path(
            instance, config_number=config_number, suffix=iteration
        )
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

            # Get headers and data
            headers = lines[0].strip().split(";")
            values = lines[1].strip().split(";")

            # Replace comma by dot in numeric values
            values = [v.replace(",", ".") for v in values]

            # Convert numeric fields if needed (optional)
            def try_cast(value):
                try:
                    return float(value)
                except ValueError:
                    return (
                        value.lower() == "true"
                        if value.lower() in ["true", "false"]
                        else value
                    )

            # Build dictionary
            data = {key: try_cast(val) for key, val in zip(headers, values)}

        return data

    def compare_arcs(
        self,
        arc_1: List[Tuple],
        arc_2: List[Tuple],
    ) -> List[Tuple]:
        """
        Compare arcs from the original configuration with those from a custom configuration.

        Args:
            arc_1: List of tuples representing the original arcs
            arc_2: List of tuples representing the custom arcs

        Returns:
            Boolean indicating whether the arcs are the same
        """
        arc_1 = load_set_arc(arc_1)
        arc_2 = load_set_arc(arc_2)

        # Find the difference between the two sets of arcs
        diff_arcs = arc_1.symmetric_difference(arc_2)
        return len(diff_arcs) == 0

    def delete_intermediate_files(
        self,
        instance: int,
        iter: int,
        config_number: str,
    ) -> None:
        """
        Delete intermediate files for a given instance and configuration.

        Args:
            instance: Instance number
            config_number: Configuration identifier
            suffix: File suffix
        """

        for suffix in range(1, int(iter)):

            arcs_path = self.get_arcs_path(instance, config_number, suffix)
            results_path = self.get_results_path(instance, config_number, suffix)
            cost_path = self.get_cost_path(instance, config_number, suffix)

            if arcs_path.exists():
                arcs_path.unlink()
            if results_path.exists():
                results_path.unlink()
            if cost_path.exists():
                print("Deleting cost file:", cost_path)
                cost_path.unlink()
