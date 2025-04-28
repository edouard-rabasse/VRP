# train.py
import os, sys
import torch
from src.data_loader_mask    import load_data_train_test
from src.models              import load_model
from src.transform           import image_transform_train, image_transform_test, mask_transform
from src.train_functions     import train
from src.evaluation          import get_confusion_matrix

def main(cfg_path: str):
    # ── load config ─────────────────────────────────────────────────────────
    sys.path.append(os.path.dirname(cfg_path))
    cfg = __import__(os.path.basename(cfg_path).replace('.py',''))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}, CUDA={torch.cuda.is_available()}")

    # ── build model ──────────────────────────────────────────────────────────
    model = load_model(cfg.model_name, device, cfg)
    print(f"[Train] Loaded model: {cfg.model_name}")

    # ── data loaders ────────────────────────────────────────────────────────
    train_loader, test_loader = load_data_train_test(
        train_original_path=cfg.train_original_path,
        test_original_path =cfg.test_original_path,
        train_modified_path=cfg.train_modified_path,
        test_modified_path =cfg.test_modified_path,
        mask_path_train    =cfg.train_mask_path,
        mask_path_test     =cfg.test_mask_path,
        batch_size         =cfg.batch_size,
        image_transform_train = image_transform_train(cfg.image_size),
        image_transform_test  = image_transform_test(cfg.image_size),
        mask_transform_train  = mask_transform(cfg.mask_shape),
        mask_transform_test   = mask_transform(cfg.mask_shape),
        # num_workers        = os.cpu_count(),
        num_workers=0
    )
    print(f"[Train] Data loaded: {len(train_loader.dataset)} train / {len(test_loader.dataset)} test")

    
    results = train(
        model_name    =cfg.model_name,
        model         =model,
        train_loader  =train_loader,
        test_loader   =test_loader,
        num_epochs    =cfg.MODEL_PARAMS["epochs"],
        device        =device,
        learning_rate =cfg.MODEL_PARAMS["learning_rate"],
        cfg           =cfg
    )
    # confusion
    cm = get_confusion_matrix(model, test_loader, device=device)
    results.append(f"\nConfusion matrix:\n{cm}")

    # save results
    os.makedirs("results", exist_ok=True)
    out = f"results/{cfg.model_name}_{cfg.cfg_number}.txt"
    with open(out, "w") as f:
        f.write("\n".join(results))
    print(f"[Train] Results written to {out}")

    # ── save model ──────────────────────────────────────────────────────────
    if cfg.save_model:
        from src.utils import save_model
        os.makedirs(os.path.dirname(cfg.weight_path), exist_ok=True)
        save_model(model, cfg.weight_path)
        print(f"[Train] Model saved at {cfg.weight_path}")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv)>1 else "config/cfg_deit.py"
    main(config_file)
