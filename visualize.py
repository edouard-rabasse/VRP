# visualize.py : this script is used to visualize the heatmap and the overlay

import os, sys
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms

from src.models import load_model
from src.transform import image_transform_test, mask_transform, denormalize
from src.visualization import get_heatmap, show_mask_on_image
from src.graph import get_arc_name, get_coordinates_name, read_arcs, read_coordinates
from src.graph.reverse_heatmap import reverse_heatmap
from evaluate_seg import compute_bce_with_logits_mask
import hydra
from omegaconf import DictConfig, OmegaConf


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
        number = int(fname.split(".")[0].split("_")[1])

        # ── load images & transforms ────────────────────────────────────────
        orig_p = os.path.join(cfg.data.test_original_path, fname)
        mask_p = os.path.join(cfg.data.test_mask_path, fname)

        img = Image.open(orig_p).convert("RGB")
        mask = Image.open(mask_p).convert("L")

        t_img = image_transform_test(cfg.image_size)(img).unsqueeze(0).to(device)
        t_den = denormalize(t_img.squeeze(0).cpu())
        print("[Debug] image mode", img.mode)

        mask = TF.resize(
            mask,
            (t_den.shape[2], t_den.shape[1]),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        mask = mask_transform(size=cfg.image_size)(mask)

        # ── heatmap & overlay ────────────────────────────────────────────────
        hm = get_heatmap(
            cfg.heatmap.method, model, t_img, cfg.heatmap.args, device=device
        )
        print("[Debug] Heatmap shape:", hm.shape)
        overlay = show_mask_on_image(
            mask, hm, alpha=0.5, interpolation=cv2.INTER_NEAREST
        )

        # ── save ───────────────────────────────────────────────────────────────
        out_p = os.path.join(output_dir, fname)
        # out_p2 = os.path.join(output_dir, f"overlay_{fname}")
        # cv2.imwrite(out_p, overlay)
        Image.fromarray(overlay).save(out_p)
        print(f"[Viz] Saved overlay to {out_p}")

        # ── reverse heatmap ────────────────────────────────────────────────────
        coordinates_dir = cfg.arcs.coord_in_dir
        arcs_dir = cfg.arcs.arcs_in_dir
        coordinates_p = os.path.join(coordinates_dir, get_coordinates_name(number))
        arcs_p = os.path.join(arcs_dir, get_arc_name(number))
        coordinates, _ = read_coordinates(coordinates_p, keep_service_time=True)
        arcs = read_arcs(arcs_p)
        arcs_with_zone, coordinates = reverse_heatmap(
            arcs=arcs,
            coordinates=coordinates,
            heatmap=hm,
            bounds=list(cfg.arcs.bounds),
            threshold=cfg.arcs.threshold,
            n_samples=cfg.arcs.n_samples,
        )
        # ── save arcs ────────────────────────────────────────────────────────────
        arcs_out_p = cfg.arcs.arcs_out_dir
        os.makedirs(arcs_out_p, exist_ok=True)
        coordinates_out_p = cfg.arcs.coord_out_dir
        os.makedirs(coordinates_out_p, exist_ok=True)
        coordinates_out_p = os.path.join(
            coordinates_out_p, get_coordinates_name(number)
        )
        arcs_out_p = os.path.join(arcs_out_p, get_arc_name(number))
        with open(arcs_out_p, "w") as f:
            for arc in arcs_with_zone:
                f.write(f"{arc[0]};{arc[1]};{arc[2]};{arc[3]};{arc[4]}\n")
        with open(coordinates_out_p, "w") as f:
            for node, coord in coordinates.items():
                f.write(f"{node},{coord[0]},{coord[1]},{coord[2]},{coord[3]}\n")

        # --- compute loss
        loss = compute_bce_with_logits_mask(hm, mask)
        running_loss += loss
    loss = running_loss / len(os.listdir(cfg.data.test_original_path))
    print(f"[Viz] Loss: {loss:.4f}")

    # print(f"[Viz] Saved arcs to {arcs_out_p}")


if __name__ == "__main__":
    main()
