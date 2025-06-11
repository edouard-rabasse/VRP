# File: src/pipeline/config.py
from hydra import initialize, compose
from pathlib import Path

# Base directory for the MSH solver
BASE_DIR = Path(__file__).parent.parent.parent / "MSH" / "MSH"

# Default Hydra overrides
DEFAULT_OVERRIDES = [
    "data=config7",
    "model=resnet",
    "model.weight_path=checkpoints/resnet_8_30_7.pth",
    "model.load=true",
]


def get_cfg(overrides: list[str] | None = None):
    """
    Charge la configuration Hydra.
    """
    with initialize(version_base=None, config_path="../../config"):
        return compose(config_name="config", overrides=overrides or DEFAULT_OVERRIDES)
