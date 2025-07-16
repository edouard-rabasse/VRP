import os
import time
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src.data_loader import load_data
from src.models import load_model
from src.train_functions import train
from src.utils.config_utils import load_selection_config


def initialize_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}, CUDA={torch.cuda.is_available()}")
    return device


def load_model_and_data(cfg, device):
    start = time.perf_counter()
    model = load_model(cfg.model.name, device, cfg.model)
    print(
        f"[Train] Loaded model: {cfg.model.name} ({time.perf_counter() - start:.2f}s)"
    )

    start = time.perf_counter()
    train_loader, test_loader = load_data(cfg)
    print(
        f"[Train] Data loaded: {len(train_loader.dataset)} train / {len(test_loader.dataset)} test ({time.perf_counter() - start:.2f}s)"
    )
    return model, train_loader, test_loader


def init_wandb(cfg, model):
    start = time.perf_counter()
    run = wandb.init(
        project="VRP",
        name=f"{cfg.model.name}_{cfg.batch_size}bs_{cfg.model_params.epochs}ep_{cfg.model_params.learning_rate}lr_cfg{cfg.data.cfg_number}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )
    run.name = f"{cfg.experiment_name}_{run.id}"
    wandb.watch(model, log="all")
    print(f"[Timer] W&B initialization took {time.perf_counter() - start:.2f}s")
    return run


def save_model_if_needed(cfg, model, name):
    if cfg.save_model:
        from src.utils.utils import save_model

        os.makedirs(os.path.dirname(cfg.model.save_path), exist_ok=True)
        save_path = f"{cfg.model.save_path}{name}.pth"
        save_model(model, save_path)
        if os.path.exists(save_path):
            print(f"[Train] ✅ Model saved at {save_path}")
        else:
            print(f"[Train] ❌ ERROR: Model was NOT saved at {save_path}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    start_total = time.perf_counter()

    device = initialize_device()

    model, train_loader, test_loader = load_model_and_data(cfg, device)

    run = init_wandb(cfg, model)

    metrics = train(
        model_name=cfg.model.name,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=cfg.model_params.epochs,
        device=device,
        learning_rate=cfg.model_params.learning_rate,
        cfg=cfg,
    )

    for epoch_metrics in metrics:
        wandb.log(epoch_metrics)
    wandb.finish()

    save_model_if_needed(cfg, model, run.name)

    print(f"[Train] Total training time: {time.perf_counter() - start_total:.2f}s")


if __name__ == "__main__":
    main()
