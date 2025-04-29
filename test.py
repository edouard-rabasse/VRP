# train.py (excerpt)
from src.utils import load_cfg
import hydra


@hydra.main(version_base=None, config_path="config", config_name="cnn")
def main(cfg):
    print(cfg.model.load)
    print(cfg.data.train_original_path)

if __name__ == "__main__":
    main()
print("test")
