# File: src/pipeline/solver_client.py
import subprocess
from pathlib import Path
from omegaconf import DictConfig


class SolverError(Exception):
    pass


class SolverClient:
    """
    Wrapper de l'appel au solveur Java (MSH + Gurobi).
    """

    def __init__(
        self,
        msh_dir: Path,
        java_lib: Path,
        program_name: str = "main.Main_customCosts",
        custom_args: list | None = None,
    ):

        self.msh_dir = msh_dir
        self.java_lib = java_lib
        self.program_name = program_name

        if custom_args is not None:
            self.custom_arguments = custom_args
        else:
            self.custom_arguments = [
                "-Xmx14000m",
                f"-Djava.library.path={self.java_lib}",
                "-cp",
                f"bin;{self.java_lib.parent / 'lib' / 'gurobi.jar'}",
            ]

        # Recompile
        self.recompile_java_files()

    def recompile_java_files(self) -> None:
        cmd = [
            "javac",
            "-cp",
            f"bin;{self.java_lib.parent / 'lib' / 'gurobi.jar'}",
            "src/main/*.java",
            "-d",
            "bin",
        ]
        print(f"[Debug] Running command: {' '.join(cmd)} from {self.msh_dir}")
        result = subprocess.run(cmd, cwd=self.msh_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise SolverError(
                f"Java compilation failed ({result.returncode})\n{result.stderr}"
            )
        return

    def run(
        self,
        instance: int,
        config_name: str,
        arc_suffix: str = "1",
        timeout: int = 300,
    ) -> None:
        cmd = [
            "java",
            *self.custom_arguments,
            self.program_name,
            f"Coordinates_{instance}.txt",
            f"Costs_{instance}_{arc_suffix}.txt",
            config_name,
            f"Arcs_{instance}_{arc_suffix}.txt",
            f"{arc_suffix}",
        ]
        print(f"[Debug] Running command: {' '.join(cmd)} from {self.msh_dir}")
        result = subprocess.run(
            cmd,
            cwd=self.msh_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise SolverError(f"Solver failed ({result.returncode})\n{result.stderr}")
        return
