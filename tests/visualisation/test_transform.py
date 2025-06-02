import torch
import numpy as np
from PIL import Image
from src.transform import (
    denormalize,
    image_transform_train,
    image_transform_test,
    MaxPoolResize,
    mask_transform,
)


def test_denormalize_shape_and_range():
    tensor = torch.randn(3, 224, 224)
    norm = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    denorm = denormalize(norm)
    assert denorm.shape == (3, 224, 224)
    assert (denorm >= 0).all() and (denorm <= 1.5).all()  # possible range after unnorm


def test_image_transform_train_and_test():
    pil_image = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype(np.uint8))

    for transform_fn in [image_transform_train, image_transform_test]:
        transform = transform_fn()
        tensor = transform(pil_image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert abs(tensor.mean()) < 2  # Rough check after normalization


def test_maxpoolresize_single_channel():
    mask = torch.zeros(1, 20, 20)
    mask[:, 5:15, 5:15] = 1.0
    resizer = MaxPoolResize(size=(10, 10))
    resized = resizer(mask)
    assert resized.shape == (1, 10, 10)
    assert resized.max() == 1.0
    assert resized.min() == 0.0


def test_mask_transform_output():
    mask_array = (np.random.rand(32, 32) > 0.5).astype(np.float32)
    pil_mask = Image.fromarray((mask_array * 255).astype(np.uint8))
    transform = mask_transform((10, 10))
    transformed = transform(pil_mask)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (1, 10, 10)
