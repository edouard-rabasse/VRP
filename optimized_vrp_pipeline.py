# File: src/main.py
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

    with initialize(version_base=None, config_path="config"):
        # Combine CLI + default overrides
        full_overrides = DEFAULT_OVERRIDES + cli_overrides
        cfg = compose(config_name="config", overrides=full_overrides)

        threshold = float(cfg.threshold)
        walking = int(cfg.walking)
        multiplier = int(cfg.multiplier)

        pipeline = OptimizedVRPPipeline(cfg)
        pipeline.run_optimized_pipeline(
            walking=walking,
            multiplier=multiplier,
            threshold=threshold,
            numbers=range(9, 40),
            max_iter=20,
            output_dir="output",
        )


if __name__ == "__main__":
    main()
