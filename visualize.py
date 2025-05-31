# visualize.py : this script is used to visualize the heatmap and the overlay

import os
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

from src.models import load_model
from src.visualization import process_image, compute_bce_with_logits_mask
import hydra
from omegaconf import DictConfig
import time


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # ── config ────────────────────────────────────────────────────────────────
    # sys.path.append(os.path.dirname(cfg_path))
    # cfg = __import__(os.path.basename(cfg_path).replace('.py',''))

    cfg.load_model = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.model.name, device, cfg.model).eval()
    print(f"[Viz] Model loaded: {cfg.model.name}")

    output_dir = cfg.heatmap_dir
    os.makedirs(output_dir, exist_ok=True)

    running_loss = 0.0

    for fname in sorted(os.listdir(cfg.data.test_original_path)):
        if not fname.endswith(".png"):
            continue
        start = time.perf_counter()

        loss = process_image(cfg, model, fname, device)
        running_loss += loss
        print(f"[timer] {fname} took {time.perf_counter() - start:.2f}s")

    loss = running_loss / len(os.listdir(cfg.data.test_original_path))
    print(f"[Viz] Loss: {loss:.4f}")

    # print(f"[Viz] Saved arcs to {arcs_out_p}")


if __name__ == "__main__":
    main()
