import os
from PIL import Image
from torchvision import transforms
from ..transform import image_transform_test, mask_transform, denormalize
from torchvision.transforms import functional as TF


def load_and_transform_image_mask(cfg, fname: str, device):
    orig_p = os.path.join(cfg.data.test_original_path, fname)
    mask_p = os.path.join(cfg.data.test_mask_path, fname)

    img = Image.open(orig_p).convert("RGB")
    mask = Image.open(mask_p).convert("L")

    t_img = image_transform_test(cfg.image_size)(img).unsqueeze(0).to(device)
    # t_den = denormalize(t_img.squeeze(0).cpu())
    mask = TF.resize(
        mask,
        (t_img.shape[-2], t_img.shape[-1]),
        interpolation=transforms.InterpolationMode.NEAREST,
    )
    mask = mask_transform(size=cfg.image_size)(mask)

    return t_img, mask
