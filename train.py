# train.py : this script trains the model, all the parameters are in config.yaml and subfolders

import os
import torch
from src.data_loader import load_data
from src.models import load_model
from src.train_functions import train
from src.utils.config_utils import load_selection_config
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from evaluate_seg import compute_seg_loss_from_loader
import time


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    start_total = time.perf_counter()
    start = time.perf_counter()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}, CUDA={torch.cuda.is_available()}")
    print(f"[Timer] Device setup took {time.perf_counter() - start:.2f}s")

    # ── build model ──────────────────────────────────────────────────────────
    start = time.perf_counter()
    model = load_model(cfg.model.name, device, cfg.model)
    print(f"[Train] Loaded model: {cfg.model.name}")
    print(f"[Timer] Model loading took {time.perf_counter() - start:.2f}s")

    # ── data loaders ────────────────────────────────────────────────────────
    start = time.perf_counter()
    range = load_selection_config(cfg.data)
    train_loader, test_loader = load_data(cfg)
    print(
        f"[Train] Data loaded: {len(train_loader.dataset)} train / {len(test_loader.dataset)} test"
    )
    print(f"[Timer] Data loading took {time.perf_counter() - start:.2f}s")

    # initialise W&B en lui passant tout le cfg Hydra
    start = time.perf_counter()
    run = wandb.init(
        project="VRP",
        name=f"{cfg.model}_{cfg.batch_size}bs_{cfg.model_params.epochs}ep_{cfg.model_params.learning_rate}lr_cfg{cfg.data.cfg_number}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )
    print(f"[Timer] W&B initialisation took {time.perf_counter() - start:.2f}s")
    # renomme avec l’ID pour plus de lisibilité
    run.name = f"{cfg.experiment_name}_{run.id}"
    # log des gradients / poids
    wandb.watch(model, log="all")

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
    # loss = compute_seg_loss_from_loader(
    #     test_loader, model, device, cfg.heatmap.method, cfg.heatmap.args
    # )
    # append loss to metrics

    for epoch_metrics in metrics:
        wandb.log(epoch_metrics)
    # wandb.log({"final_seg_loss": loss})
    wandb.finish()

    # ── save model ──────────────────────────────────────────────────────────
    if cfg.save_model:
        from src.utils.utils import save_model

        os.makedirs(os.path.dirname(cfg.model.weight_path), exist_ok=True)
        save_model(model, cfg.model.weight_path)
        if os.path.exists(cfg.model.weight_path):
            print(f"[Train] ✅ Model saved at {cfg.model.weight_path}")
        else:
            print(f"[Train] ❌ ERROR: Model was NOT saved at {cfg.model.weight_path}")


if __name__ == "__main__":
    main()
