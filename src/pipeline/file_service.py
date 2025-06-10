# File: src/pipeline/file_service.py
from pathlib import Path
from typing import List, Dict, Tuple

from src.graph import read_coordinates, read_arcs


class FileService:
    """
    Gestion centralisée des chemins et I/O de fichiers pour les instances et solutions.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def coord_path(self, instance: int) -> Path:
        # TODO: modify
        coord_path = (
            self.base_dir / "instancesCustomCosts" / f"Coordinates_{instance}.txt"
        )
        if not coord_path.exists():
            raise FileNotFoundError(f"The file {coord_path} does not exist.")
        return coord_path

    def arc_path(self, instance: int, config: str = "1", suffix: str = "1") -> Path:
        return (
            self.base_dir
            / "results"
            / f"configuration{config}"
            / f"Arcs_{instance}_{suffix}.txt"
        )

    def solution_arc_path(
        self, instance: int, suffix: str = "1", config: str = "CustomCosts"
    ) -> Path:
        return (
            self.base_dir
            / "results"
            / f"configuration{config}"
            / f"Arcs_{instance}_{suffix}.txt"
        )

    def read_coordinates(
        self, instance: int, type="original", keep_service_time=False
    ) -> Tuple[dict, int]:
        return read_coordinates(str(self.coord_path(instance)), type, keep_service_time)

    def read_arcs(
        self, instance: int, config="1", suffix: str = "1", type="original"
    ) -> List[Tuple]:
        return read_arcs(
            str(self.arc_path(instance, config=config, suffix=suffix)), type
        )

    def save_arcs(
        self, instance: int, arcs: List[Tuple], config: str = "1", suffix: str = "1"
    ) -> None:
        path = self.arc_path(instance, config=config, suffix=suffix)
        with open(path, "w") as f:
            for arc in arcs:
                f.write(f"{arc[0]};{arc[1]};{arc[2]};{arc[3]};{arc[4]}\n")
