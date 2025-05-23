import os
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models import load_model
from src.transform import image_transform_test, mask_transform, denormalize
from src.visualization import get_heatmap, show_mask_on_image
from src.graph.reverse_heatmap import (
    reverse_heatmap,
    get_arc_name,
    get_coordinates_name,
    read_arcs,
    read_coordinates,
)
from src.utils.config_utils import load_selection_config
from src.data_loader_mask import load_data


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cfg.load_model = True
    model = load_model(cfg.model.name, device, cfg.model).eval()
    os.makedirs(f"output/{cfg.heatmap.method}_{cfg.model.name}", exist_ok=True)

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
        return_filenames=True,
    )

    for imgs, labels, masks, fnames in test_loader:
        # imgs: [B,3,H,W], masks: [B,1,H,W], filenames: list of strings
        imgs = imgs.to(device)
        # ── batched heatmap ────────────────────────────────────────────────────

        # heatmaps: tensor [B, H, W] or [B,1,H,W] depending on implementation

        # ── per-sample postprocessing ──────────────────────────────────────────
        for img_tensor, mask_tensor, fname in zip(imgs, masks, fnames):
            hm = get_heatmap(
                cfg.heatmap.method,
                model,
                img_tensor.unsqueeze(0),
                cfg.heatmap.args,
                device=device,
            )
            # denormalize the image back to uint8 H×W×3
            # ensure mask & heatmap are numpy 2D arrays
            mask_np = mask_tensor.cpu().numpy()
            print("heatmap shape:", hm.shape)
            hm_np = hm

            overlay = show_mask_on_image(
                mask_np, hm_np, alpha=0.5, interpolation=cv2.INTER_NEAREST
            )

            out_path = os.path.join(
                f"output/{cfg.heatmap.method}_{cfg.model.name}", fname
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(overlay).save(out_path)

            # ── reverse heatmap per-sample ────────────────────────────────────
            # extract sample index from filename (must match your naming)
            number = int(os.path.splitext(fname)[0].split("_")[1])
            coord_in = os.path.join(cfg.arcs.coord_in_dir, get_coordinates_name(number))
            arcs_in = os.path.join(cfg.arcs.arcs_in_dir, get_arc_name(number))
            coords, _ = read_coordinates(coord_in)
            arcs = read_arcs(arcs_in)

            arcs_with_zone, coords_out = reverse_heatmap(
                arcs=arcs,
                coordinates=coords,
                heatmap=hm_np,
                bounds=list(cfg.arcs.bounds),
                threshold=cfg.arcs.threshold,
                n_samples=cfg.arcs.n_samples,
            )

            # save reverse-heatmap outputs
            os.makedirs(cfg.arcs.arcs_out_dir, exist_ok=True)
            os.makedirs(cfg.arcs.coord_out_dir, exist_ok=True)
            arcs_out_p = os.path.join(cfg.arcs.arcs_out_dir, get_arc_name(number))
            coord_out_p = os.path.join(
                cfg.arcs.coord_out_dir, get_coordinates_name(number)
            )

            with open(arcs_out_p, "w") as f:
                for arc in arcs_with_zone:
                    f.write(f"{arc[0]};{arc[1]};{arc[2]};{arc[3]};{arc[4]}\n")
            with open(coord_out_p, "w") as f:
                for node, c in coords_out.items():
                    f.write(f"{node},{c[0]},{c[1]},{c[2]}\n")

    print("[Viz] Done.")


if __name__ == "__main__":
    main()
