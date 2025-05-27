# train.py : this script trains the model, all the parameters are in config.yaml and subfolders

import os, sys
import torch
from src.data_loader import load_data
from src.models import load_model
from src.transform import image_transform_train, image_transform_test, mask_transform
from src.train_functions import train
from src.evaluation import get_confusion_matrix
from src.utils.config_utils import load_selection_config
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from evaluate_seg import compute_seg_loss_from_loader


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}, CUDA={torch.cuda.is_available()}")

    # ── build model ──────────────────────────────────────────────────────────
    model = load_model(cfg.model.name, device, cfg.model)
    print(f"[Train] Loaded model: {cfg.model.name}")

    # ── data loaders ────────────────────────────────────────────────────────
    print(cfg.data.selection.value)
    print("a")
    range = load_selection_config(cfg.data)
    train_loader, test_loader = load_data(cfg)
    print(
        f"[Train] Data loaded: {len(train_loader.dataset)} train / {len(test_loader.dataset)} test"
    )

    # initialise W&B en lui passant tout le cfg Hydra
    run = wandb.init(
        project="VRP",
        name=f"{cfg.model}_{cfg.batch_size}bs_{cfg.model_params.epochs}ep_{cfg.model_params.learning_rate}lr_cfg{cfg.data.cfg_number}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
    )
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
    loss = compute_seg_loss_from_loader(
        test_loader, model, device, cfg.heatmap.method, cfg.heatmap.args
    )
    # append loss to metrics

    for epoch_metrics in metrics:
        wandb.log(epoch_metrics)
    wandb.log({"final_seg_loss": loss})
    wandb.finish()

    # confusion
    # cm = get_confusion_matrix(model, test_loader, device=device)
    # results.append(f"\nConfusion matrix:\n{cm}")

    # # save results
    # os.makedirs("results", exist_ok=True)
    # out = f"results/{cfg.model.name}_{cfg.data.cfg_number}_{cfg.model.params.learning_rate}.txt"
    # with open(out, "w") as f:
    #     f.write("\n".join(results))
    # print(f"[Train] Results written to {out}")

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
