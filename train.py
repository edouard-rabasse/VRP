# train.py
import os, sys
import torch
from src.data_loader_mask    import load_data_train_test
from src.models              import load_model
from src.transform           import image_transform_train, image_transform_test, mask_transform
from src.train_functions     import train
from src.evaluation          import get_confusion_matrix
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf



@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}, CUDA={torch.cuda.is_available()}")
    


    # ── build model ──────────────────────────────────────────────────────────
    model = load_model(cfg.model.name, device, cfg.model)
    print(f"[Train] Loaded model: {cfg.model.name}")

    # ── data loaders ────────────────────────────────────────────────────────
    train_loader, test_loader = load_data_train_test(
        train_original_path=cfg.data.train_original_path,
        test_original_path =cfg.data.test_original_path,
        train_modified_path=cfg.data.train_modified_path,
        test_modified_path =cfg.data.test_modified_path,
        mask_path_train    =cfg.data.train_mask_path,
        mask_path_test     =cfg.data.test_mask_path,
        batch_size         =cfg.batch_size,
        image_transform_train = image_transform_train(tuple(cfg.image_size)),
        image_transform_test  = image_transform_test(tuple(cfg.image_size)),
        mask_transform_train  = mask_transform(tuple(cfg.mask_shape)),
        mask_transform_test   = mask_transform(tuple(cfg.mask_shape)),
        # num_workers        = os.cpu_count(),
        num_workers=2
    )
    print(f"[Train] Data loaded: {len(train_loader.dataset)} train / {len(test_loader.dataset)} test")

        # initialise W&B en lui passant tout le cfg Hydra
    run = wandb.init(
        project="VRP",
        name=f"{cfg.experiment_name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )
    # renomme avec l’ID pour plus de lisibilité
    run.name = f"{cfg.experiment_name}_{run.id}"
    # log des gradients / poids
    wandb.watch(model, log="all")


    
    metrics = train(
        model_name    =cfg.model.name,
        model         =model,
        train_loader  =train_loader,
        test_loader   =test_loader,
        num_epochs    =cfg.model_params.epochs,
        device        =device,
        learning_rate =cfg.model_params.learning_rate,
        cfg           =cfg
    )
    for epoch_metrics in metrics:
        wandb.log(epoch_metrics)
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
        from src.utils import save_model
        os.makedirs(os.path.dirname(cfg.model.weight_path), exist_ok=True)
        save_model(model, cfg.model.weight_path)
        if os.path.exists(cfg.model.weight_path):
            print(f"[Train] ✅ Model saved at {cfg.model.weight_path}")
        else:
            print(f"[Train] ❌ ERROR: Model was NOT saved at {cfg.model.weight_path}")

if __name__ == "__main__":
    main()
