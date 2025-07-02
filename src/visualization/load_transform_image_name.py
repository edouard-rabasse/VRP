## This function loads an image and its corresponding mask, applies transformations,

import os
from PIL import Image
from torchvision import transforms
from ..transform import image_transform_test, mask_transform, denormalize
from torchvision.transforms import functional as TF


def load_and_transform_image_mask(cfg, image_path, mask_path, fname: str, device: str):
    """Load and transform an original image and its corresponding mask.

    Args:
        cfg (DictConfig): Configuration object containing paths and settings.
        fname (str): Filename of the image to load.
        device (str): Device to which the tensors should be moved.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed image tensor and mask tensor.
    """
    # img = Image.open(image_path).convert("RGB")
    # mask = Image.open(mask_path).convert("L")

    # img = Image.open(orig_p).convert("RGB")
    # mask = Image.open(mask_p).convert("L")

    # t_img = image_transform_test(cfg.image_size)(img).unsqueeze(0).to(device)
    # # t_den = denormalize(t_img.squeeze(0).cpu())
    # mask = TF.resize(
    #     mask,
    #     (t_img.shape[-2], t_img.shape[-1]),
    #     interpolation=transforms.InterpolationMode.NEAREST,
    # )
    # mask = mask_transform(size=cfg.image_size)(mask)
    # # mask = mask_transform(cfg.mask_shape)(mask)
    t_img = load_transform_image(cfg, image_path, fname, device)
    mask = load_transform_mask(cfg, mask_path, fname, device)

    return t_img, mask


def load_transform_image(cfg, image_path: str, fname: str, device: str):
    """Load and transform an image.

    Args:
        cfg (DictConfig): Configuration object containing paths and settings.
        image_path (str): Path to the image file.
        fname (str): Filename of the image to load.
        device (str): Device to which the tensor should be moved.

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    img = Image.open(os.path.join(image_path, fname)).convert("RGB")
    t_img = image_transform_test(cfg.image_size)(img).unsqueeze(0).to(device)
    return t_img


def load_transform_mask(cfg, mask_path: str, fname: str, device: str):
    """Load and transform a mask.

    Args:
        cfg (DictConfig): Configuration object containing paths and settings.
        mask_path (str): Path to the mask file.
        fname (str): Filename of the mask to load.
        device (str): Device to which the tensor should be moved.

    Returns:
        torch.Tensor: Transformed mask tensor.
    """
    mask = Image.open(os.path.join(mask_path, fname)).convert("L")
    mask = TF.resize(
        mask,
        (cfg.image_size, cfg.image_size),
        interpolation=transforms.InterpolationMode.NEAREST,
    )
    mask = mask_transform(size=cfg.image_size)(mask)
    # mask = mask_transform(cfg.mask_shape)(mask)

    return mask
