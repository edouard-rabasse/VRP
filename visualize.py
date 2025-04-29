# visualize.py
import os, sys
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

from src.models          import load_model
from src.transform       import image_transform_test, mask_transform, denormalize
from src.visualization   import get_heatmap, show_mask_on_image
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # ── config ────────────────────────────────────────────────────────────────
    # sys.path.append(os.path.dirname(cfg_path))
    # cfg = __import__(os.path.basename(cfg_path).replace('.py',''))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.model.name, device, cfg.model).eval()
    print(f"[Viz] Model loaded: {cfg.model.name}")

    output_dir = f"output/{cfg.heatmap.method}_{cfg.model.name}"
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(cfg.data.test_modified_path)):
        if not fname.endswith('.png'): continue

        # ── load images & transforms ────────────────────────────────────────
        orig_p = os.path.join(cfg.data.test_original_path, fname)
        mask_p = os.path.join(cfg.data.test_mask_path,     fname)

        img   = Image.open(orig_p).convert("RGB")
        mask  = Image.open(mask_p).convert("L")

        t_img = image_transform_test(cfg.image_size)(img).unsqueeze(0).to(device)
        t_den = denormalize(t_img.squeeze(0).cpu())
        print("[Debug] image mode", img.mode)

        mask = TF.resize(mask, (t_den.shape[2], t_den.shape[1]), interpolation=transforms.InterpolationMode.NEAREST)
        mask = mask_transform(size=cfg.image_size)(mask)

        # ── heatmap & overlay ────────────────────────────────────────────────
        hm      = get_heatmap(cfg.heatmap.method, model, t_img, cfg.heatmap.args, device=device)
        print("[Debug] Heatmap shape:", hm.shape)
        overlay = show_mask_on_image(mask, hm, alpha=0.5)

        # ── save ───────────────────────────────────────────────────────────────
        out_p = os.path.join(output_dir, fname)
        # out_p2 = os.path.join(output_dir, f"overlay_{fname}")
        # cv2.imwrite(out_p, overlay)
        Image.fromarray(overlay).save(out_p)
        print(f"[Viz] Saved overlay to {out_p}")


if __name__ == "__main__":
    # cfg_file = sys.argv[1] if len(sys.argv)>1 else "config/cfg_deit.py"
    main()
