from pathlib import Path
from typing import List, Dict, Tuple, Optional

from src.graph import read_coordinates, read_arcs


class FileService:
    """
    Centralized file path management and I/O operations for VRP instances and solutions.

    Handles file paths, reading coordinates and arcs, and saving results for VRP instances
    """

    ""
    DEFAULT_INSTANCE_FOLDER = "instancesCustomCosts"
    DEFAULT_RESULTS_FOLDER = "results"
    DEFAULT_CONFIG = "1"
    DEFAULT_SUFFIX = "1"
    ORIGINAL_CONFIG = "1"
    CUSTOM_COSTS_CONFIG = "CustomCosts"

    def __init__(
        self,
        base_dir: Path,
        instance_folder: str = DEFAULT_INSTANCE_FOLDER,
        results_folder: str = DEFAULT_RESULTS_FOLDER,
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

    def get_cost_path(self, instance: int, suffix: str) -> Path:
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
            / f"CostsAnalysis_{instance}_{suffix}.txt"
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
        config_number: str = DEFAULT_CONFIG,
        suffix: str = DEFAULT_SUFFIX,
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
        org_cfg_number: str = ORIGINAL_CONFIG,
        cfg_number: str = CUSTOM_COSTS_CONFIG,
    ) -> None:
        """Creates a copy of the results file for a given instance."""
        original_path = self.get_arcs_path(
            instance, org_cfg_number, self.DEFAULT_SUFFIX
        )
        copy_path = self.get_arcs_path(instance, cfg_number, self.DEFAULT_SUFFIX)

        if not original_path.exists():
            raise FileNotFoundError(f"Original file not found: {original_path}")

        copy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(original_path, "r", encoding="utf-8") as original_file:
            with open(copy_path, "w", encoding="utf-8") as copy_file:
                copy_file.write(original_file.read())
