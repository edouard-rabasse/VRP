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



def image_transform_train(size=(224,224), mean=mean, std=std):
    """
    Transform for training images.
    Args:
        image (PIL Image or numpy array): Input image.
    Returns:
        torch.Tensor: Transformed image tensor.
    """
    return transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(size),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),  # Random rotation for augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std),
    ])

def image_transform_test(size=(224,224), mean=mean, std=std):
    """
    Transform for testing images.
    Args:
        image (PIL Image or numpy array): Input image.
    Returns:
        torch.Tensor: Transformed image tensor.
    """
    return transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std),
    ])

def mask_transform(size=(224,224)):
    """
    Transform for masks.
    Args:
        mask (numpy array): Input mask.
    Returns:
        torch.Tensor: Transformed mask tensor.
    """
    def max_pool_resize(img: torch.Tensor) -> torch.Tensor:
        """
        img : Tensor uint8 [C, H, W] (PILToTensor ne normalise pas)
        """
        img = img.float()                         # adaptive_pool attend float
        img = F.adaptive_max_pool2d(img, size)
        # --- re-binarisation pour le cas mono-canal ---
        if img.size(0) == 1:
            img = (img > 0).float()
        return img

    return transforms.Compose([
        transforms.PILToTensor(),     # 0-255, uint8, shape (C,H,W)
        transforms.Lambda(max_pool_resize)
    ])
