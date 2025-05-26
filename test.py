import os
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models import load_model
from src.transform import image_transform_test, mask_transform

from src.utils.config_utils import load_selection_config
from src.data_loader_mask import load_data
from evaluate_seg import compute_seg_loss_from_loader


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.load_model = True
    model = load_model(cfg.model.name, device, cfg.model).eval()
    # ── build your test loader ────────────────────────────────────────────────
    sel_range = load_selection_config(cfg.data)
    test_loader = load_data(
        original_path=cfg.data.test_original_path,
        modified_path=cfg.data.test_modified_path,
        mask_path=cfg.data.test_mask_path,
        batch_size=cfg.batch_size,
        img_transform=image_transform_test(tuple(cfg.image_size)),
        mask_transform=mask_transform(tuple(cfg.mask_shape)),
        num_workers=2,
        range=sel_range,
        return_filenames=False,
    )
    loss = compute_seg_loss_from_loader(
        test_loader, model, device, cfg.heatmap.method, cfg.heatmap.args
    )
    print(f"Test loss: {loss:.4f}")


if __name__ == "__main__":
    main()
