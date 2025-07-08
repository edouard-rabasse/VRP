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
    print(cfg.model)


if __name__ == "__main__":
    main()
