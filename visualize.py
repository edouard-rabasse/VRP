# visualize.py
import os, sys
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF

from src.models          import load_model
from src.transform       import image_transform_test, mask_transform, denormalize
from src.visualization   import get_heatmap, show_mask_on_image

def main(cfg_path: str):
    # ── config ────────────────────────────────────────────────────────────────
    sys.path.append(os.path.dirname(cfg_path))
    cfg = __import__(os.path.basename(cfg_path).replace('.py',''))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.model_name, device, cfg).eval()
    print(f"[Viz] Model loaded: {cfg.model_name}")

    output_dir = f"output/{cfg.method}_{cfg.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(cfg.test_modified_path)):
        if not fname.endswith('.png'): continue

        # ── load images & transforms ────────────────────────────────────────
        orig_p = os.path.join(cfg.test_original_path, fname)
        mask_p = os.path.join(cfg.test_mask_path,     fname)

        img   = Image.open(orig_p).convert("RGB")
        mask  = Image.open(mask_p).convert("L")

        t_img = image_transform_test(cfg.image_size)(img).unsqueeze(0).to(device)
        t_den = denormalize(t_img.squeeze(0).cpu())

        # ── heatmap & overlay ────────────────────────────────────────────────
        hm      = get_heatmap(cfg.method, model, t_img, cfg.heatmap_args, device=device)
        overlay = show_mask_on_image(t_den.numpy(), hm, alpha=0.5)

        # ── save ───────────────────────────────────────────────────────────────
        out_p = os.path.join(output_dir, fname)
        cv2.imwrite(out_p, overlay)
        print(f"[Viz] Saved overlay to {out_p}")

if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv)>1 else "config/cfg_deit.py"
    main(cfg_file)
