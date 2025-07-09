# File: src/main.py
from src.pipeline.optimized_pipeline import OptimizedVRPPipeline
from src.pipeline.config import override_java_param
from time import time
import hydra
from omegaconf import DictConfig
from hydra import compose
import sys

DEFAULT_OVERRIDES = [
    "data=config7",
    "model=resnet",
    "model.weight_path=checkpoints/resnet_8_30_7.pth",
    "model.load=true",
]


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    cfg = compose(config_name="config", overrides=DEFAULT_OVERRIDES)
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.00002
    walking = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    multiplier = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    pipeline = OptimizedVRPPipeline(cfg)
    # Override Java parameters for the MSH solver
    pipeline.run_optimized_pipeline(
        walking=walking,
        multiplier=multiplier,
        threshold=threshold,
        numbers=range(1, 40),  # Adjust the range as needed
        max_iter=10,
        output_dir="output",
    )


if __name__ == "__main__":
    main()
