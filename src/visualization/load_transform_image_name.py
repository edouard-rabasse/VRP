## This function loads an image and its corresponding mask, applies transformations,

import os
from PIL import Image
from torchvision import transforms
from ..transform import image_transform_test, mask_transform, denormalize
from torchvision.transforms import functional as TF


def load_and_transform_image_mask(cfg, fname: str, device: str):
    """Load and transform an original image and its corresponding mask.

    Args:
        cfg (DictConfig): Configuration object containing paths and settings.
        fname (str): Filename of the image to load.
        device (str): Device to which the tensors should be moved.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed image tensor and mask tensor.
    """
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
    # mask = mask_transform(cfg.mask_shape)(mask)

    return t_img, mask
