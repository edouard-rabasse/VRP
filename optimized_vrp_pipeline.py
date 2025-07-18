# File: src/main.py
import os
from omegaconf import OmegaConf
from src.pipeline.optimized_pipeline import OptimizedVRPPipeline
from src.pipeline.config import override_java_param
from time import time
import hydra
from hydra import initialize
from omegaconf import DictConfig
from hydra import compose
import sys

DEFAULT_OVERRIDES = [
    "data=config7",
    "model=resnet",
    "model.weight_path=checkpoints/resnet_8_30_7.pth",
    "model.load=true",
]


def main():
    cli_overrides = sys.argv[1:]

    OmegaConf.register_new_resolver("env", lambda var: os.environ.get(var, ""))

    with initialize(version_base=None, config_path="config"):
        # Combine CLI + default overrides
        full_overrides = DEFAULT_OVERRIDES + cli_overrides
        cfg = compose(config_name="config", overrides=full_overrides)

        # stop

        threshold = float(cfg.solver.threshold)
        walking = int(cfg.solver.walking)
        multiplier = int(cfg.solver.multiplier)

        pipeline = OptimizedVRPPipeline(cfg)
        pipeline.run_optimized_pipeline(
            walking=walking,
            multiplier=multiplier,
            threshold=threshold,
            numbers=range(1001, 1100),
            max_iter=50,
            output_dir="output",
        )


if __name__ == "__main__":
    main()
