# File: src/main.py
from src.pipeline.optimized_pipeline import OptimizedVRPPipeline
from src.pipeline.config import override_java_param
from time import time
import hydra
from omegaconf import DictConfig
from hydra import compose

DEFAULT_OVERRIDES = [
    "data=config7",
    "model=resnet",
    "model.weight_path=checkpoints/resnet_8_30_7.pth",
    "model.load=true",
]


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    cfg = compose(config_name="config", overrides=DEFAULT_OVERRIDES)
    pipeline = OptimizedVRPPipeline()

    for threshold in [0.00002]:
        for walking in [10]:
            for multiplier in [1, 0.5, 0.1]:
                # Override Java parameters for the MSH solver
                pipeline.run_optimized_pipeline(
                    walking=walking,
                    multiplier=multiplier,
                    threshold=threshold,
                    numbers=range(1001, 1100),  # Adjust the range as needed
                    max_iter=100,
                    output_dir="output",
                )
