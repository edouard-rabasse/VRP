# transform.py : No longer in use.

import torch
from torchvision import transforms
import torch.nn.functional as F

size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def denormalize(tensor, mean=mean, std=std):
    """
    Denormalize a tensor image using mean and std lists.

    Args:
        tensor (torch.Tensor): Normalized image tensor of shape (C, H, W) or (N, C, H, W).
        mean (list or tuple): Sequence of means for each channel.
        std (list or tuple): Sequence of standard deviations for each channel.
    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    # If tensor has batch dimension (N, C, H, W)
    if tensor.ndim == 4:
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    else:
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)

    tensor = tensor * std + mean
    return tensor


def image_transform_train(size=(224, 224), mean=mean, std=std):
    """
    Transform for training images.
    Args:
        image (PIL Image): Input image.
    Returns:
        torch.Tensor: Transformed image tensor.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.NEAREST_EXACT
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def image_transform_no_normalize(size=(224, 224)):
    """
    Transform for images without normalization.
    Args:
        image (PIL Image): Input image.
    Returns:

        torch.Tensor: Transformed image tensor.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.NEAREST_EXACT
            ),
            transforms.ToTensor(),
        ]
    )


def image_transform_test(size=(224, 224), mean=mean, std=std):
    """
    Transform for testing images.
    Args:
        image (PIL Image): Input image.
    Returns:
        torch.Tensor: Transformed image tensor.
    """
    return transforms.Compose(
        [
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.NEAREST_EXACT
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class MaxPoolResize:
    def __init__(self, size=(10, 10)):
        self.size = size

    def __call__(self, mask):
        # mask: Tensor[C,H,W]
        mask = mask.float()
        mask = F.adaptive_max_pool2d(mask, self.size)
        # --- re-binarisation pour le cas mono-canal ---
        if mask.size(0) == 1:
            mask = (mask > 0).float()

        return mask


class ProportionalThresholdResize:
    def __init__(self, size=(10, 10), threshold=0.1):
        """
        Args:
            size (tuple): output spatial size (H_out, W_out)
            threshold (float): proportion threshold at which output reaches 1.0
        """
        self.size = size
        self.threshold = threshold

    def __call__(self, mask):
        """
        Args:
            mask (Tensor[C, H, W]): binary mask with values 0 or 1

        Returns:
            Tensor[C, H_out, W_out]: resized mask with linear interpolation up to threshold
        """
        mask = mask.float()
        proportions = F.adaptive_avg_pool2d(mask, self.size)

        # Linear interpolation: scale up proportion to reach 1.0 at threshold
        scaled = proportions / (self.threshold + 1e-6)
        scaled = torch.clamp(scaled, max=1.0)  # Prevent values > 1.0

        return scaled


def mask_transform(size=(224, 224)):
    return transforms.Compose(
        [transforms.ToTensor(), ProportionalThresholdResize(size)]
    )
