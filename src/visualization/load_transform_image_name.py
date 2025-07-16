"""
Image and mask loading utilities with transformations.
Provides robust loading and transformation of images and masks for visualization.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Union
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from omegaconf import DictConfig

from ..transform import image_transform_test, mask_transform


class ImageLoadError(Exception):
    """Custom exception for image loading errors."""

    pass


class ImageMaskLoader:
    """Handles loading and transformation of images and masks."""

    def __init__(self, cfg: DictConfig, device: Union[str, torch.device]):
        """
        Initialize the loader with configuration and device.

        Args:
            cfg: Configuration object containing paths and settings
            device: Device to move tensors to
        """
        self.cfg = cfg
        self.device = torch.device(device) if isinstance(device, str) else device
        self.image_size = cfg.image_size
        self.mask_shape = cfg.mask_shape

        # Pre-compute transforms for efficiency
        self._image_transform = image_transform_test(self.image_size)
        self._mask_transform = mask_transform(size=self.image_size)

    def load_image_and_mask(
        self, image_path: Union[str, Path], mask_path: Union[str, Path], filename: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and transform both image and mask.

        Args:
            image_path: Path to the image directory
            mask_path: Path to the mask directory
            filename: Name of the file to load

        Returns:
            Tuple of (transformed_image, transformed_mask)

        Raises:
            ImageLoadError: If files cannot be loaded or transformed
        """
        try:
            image_tensor = self.load_image(image_path, filename)
            mask_tensor = self.load_mask(mask_path, filename)
            return image_tensor, mask_tensor

        except Exception as e:
            raise ImageLoadError(
                f"Failed to load image and mask for {filename}: {str(e)}"
            ) from e

    def load_image(self, image_path: Union[str, Path], filename: str) -> torch.Tensor:
        """
        Load and transform a single image.

        Args:
            image_path: Path to the image directory
            filename: Name of the image file

        Returns:
            Transformed image tensor with shape (1, C, H, W)

        Raises:
            ImageLoadError: If image cannot be loaded or transformed
        """
        file_path = Path(image_path) / filename

        if not file_path.exists():
            raise ImageLoadError(f"Image file not found: {file_path}")

        try:
            # Load and convert to RGB
            image = Image.open(file_path).convert("RGB")

            # Apply transforms and add batch dimension
            transformed_image = (
                self._image_transform(image).unsqueeze(0).to(self.device)
            )

            return transformed_image

        except Exception as e:
            raise ImageLoadError(
                f"Failed to load/transform image {filename}: {str(e)}"
            ) from e

    def load_mask(self, mask_path: Union[str, Path], filename: str) -> torch.Tensor:
        """
        Load and transform a single mask.

        Args:
            mask_path: Path to the mask directory
            filename: Name of the mask file

        Returns:
            Transformed mask tensor

        Raises:
            ImageLoadError: If mask cannot be loaded or transformed
        """
        file_path = Path(mask_path) / filename

        if not file_path.exists():
            raise ImageLoadError(f"Mask file not found: {file_path}")

        try:
            # Load and convert to grayscale
            mask = Image.open(file_path).convert("L")

            # Resize with nearest neighbor interpolation (important for masks)
            mask = TF.resize(
                mask,
                self.image_size,
                interpolation=transforms.InterpolationMode.NEAREST,
            )

            # Apply mask transform
            transformed_mask = self._mask_transform(mask)

            return transformed_mask

        except Exception as e:
            raise ImageLoadError(
                f"Failed to load/transform mask {filename}: {str(e)}"
            ) from e

    def validate_paths(
        self, image_path: Union[str, Path], mask_path: Union[str, Path]
    ) -> Tuple[Path, Path]:
        """
        Validate that the provided paths exist.

        Args:
            image_path: Path to image directory
            mask_path: Path to mask directory

        Returns:
            Tuple of validated Path objects

        Raises:
            ImageLoadError: If paths don't exist
        """
        img_path = Path(image_path)
        msk_path = Path(mask_path)

        if not img_path.exists():
            raise ImageLoadError(f"Image directory not found: {img_path}")
        if not msk_path.exists():
            raise ImageLoadError(f"Mask directory not found: {msk_path}")

        return img_path, msk_path

    def check_file_exists(self, directory: Union[str, Path], filename: str) -> bool:
        """
        Check if a file exists in the given directory.

        Args:
            directory: Directory path
            filename: Name of the file

        Returns:
            True if file exists, False otherwise
        """
        return (Path(directory) / filename).exists()


# Convenience functions for backward compatibility
def load_and_transform_image_mask(
    cfg: DictConfig, image_path: str, mask_path: str, fname: str, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Legacy function for loading and transforming image and mask.

    Args:
        cfg: Configuration object
        image_path: Path to image directory
        mask_path: Path to mask directory
        fname: Filename
        device: Device string

    Returns:
        Tuple of (image_tensor, mask_tensor)
    """
    loader = ImageMaskLoader(cfg, device)
    return loader.load_image_and_mask(image_path, mask_path, fname)


def load_transform_image(
    cfg: DictConfig, image_path: str, fname: str, device: str
) -> torch.Tensor:
    """
    Legacy function for loading and transforming an image.

    Args:
        cfg: Configuration object
        image_path: Path to image file
        fname: Filename
        device: Device string

    Returns:
        Transformed image tensor
    """
    loader = ImageMaskLoader(cfg, device)
    return loader.load_image(image_path, fname)


def load_transform_mask(
    cfg: DictConfig, mask_path: str, fname: str, device: str
) -> torch.Tensor:
    """
    Legacy function for loading and transforming a mask.

    Args:
        cfg: Configuration object
        mask_path: Path to mask file
        fname: Filename
        device: Device string

    Returns:
        Transformed mask tensor
    """
    loader = ImageMaskLoader(cfg, device)
    return loader.load_mask(mask_path, fname)


# Context manager for batch processing
class BatchImageLoader:
    """Context manager for efficient batch loading of images."""

    def __init__(self, cfg: DictConfig, device: Union[str, torch.device]):
        self.loader = ImageMaskLoader(cfg, device)
        self._cache = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cache.clear()

    def load_batch(
        self, image_path: Union[str, Path], mask_path: Union[str, Path], filenames: list
    ) -> Tuple[list, list]:
        """
        Load a batch of images and masks.

        Args:
            image_path: Path to image directory
            mask_path: Path to mask directory
            filenames: List of filenames to load

        Returns:
            Tuple of (image_list, mask_list)
        """
        images = []
        masks = []

        for filename in filenames:
            try:
                img, mask = self.loader.load_image_and_mask(
                    image_path, mask_path, filename
                )
                images.append(img)
                masks.append(mask)
            except ImageLoadError as e:
                print(f"Warning: Skipping {filename} due to error: {e}")
                continue

        return images, masks
