"""
This module provides a registry and loader functions for various neural network models used in visual scoring and segmentation tasks.
Available Models:
    - VisualScoringModel (CNN-based)
    - MultiTaskVisualScoringModel (multi-task learning)
    - VGG-based models (classification and segmentation)
    - MultiTaskVGG (multi-task VGG)
    - ResNetScoringModel (ResNet-based)
    - DeiT-tiny (Vision Transformer)
    - SegmentVGG (VGG-based segmentation)
Functions:
    - load_model(model_name: str, device: torch.device, cfgm) -> nn.Module:
        Builds and returns a model specified by `model_name`, moves it to the given device, and optionally loads pretrained weights if specified in `cfgm`.
Model Loader Functions (internal use):
    - _load_cnn(cfgm, device): Loads a VisualScoringModel.
    - _load_deit_tiny(cfgm, device): Loads a DeiT-tiny model.
    - _load_multi_task(cfgm, device): Loads a MultiTaskVisualScoringModel.
    - _load_vgg(cfgm, device): Loads a VGG model.
    - _load_MFCN(cfgm, device): Loads a MultiTaskVGG model.
    - _load_resnet(cfgm, device): Loads a ResNetScoringModel.
    - _load_vgg_segment(cfgm, device): Loads a SegmentVGG model.
Model Registry:
    Maps string model names to their corresponding loader functions for easy instantiation.
Arguments:
    model_name (str): Name of the model to load. Must be one of the keys in the registry.
    device (torch.device): The device to which the model will be moved.
    cfgm: Configuration object containing model parameters and options, including:
        - image_size
        - kernel_size
        - mask_shape
        - freeze
        - grad_layer
        - load (bool): Whether to load pretrained weights.
        - weight_path (str): Path to the pretrained weights file.
Raises:
    ValueError: If an unknown model name is provided.
    FileNotFoundError: If pretrained weights are requested but the file does not exist.
"""

import os
import torch
import torch.nn as nn

from .VisualScoringModel import VisualScoringModel
from .MultiTaskVisualModel import MultiTaskVisualScoringModel
from .vgg import load_vgg
from .MFCN import MultiTaskVGG
from .resnet import ResNetScoringModel
from .deit_tiny import load_deit
from .vgg_segment import SegmentVGG


def _load_cnn(cfgm, device):
    H, W = cfgm.image_size
    kernel_sizes = [int(i) for i in (cfgm.kernel_size).split(",")]
    model = VisualScoringModel(input_shape=(3, H, W))
    return model


def _load_deit_tiny(cfgm, device):
    return load_deit(
        model_name="deit_tiny",
        num_classes=2,
        freeze=cfgm.freeze,
        device=device,
    )


def _load_multi_task(cfgm, device):
    H, W = cfgm.image_size
    model = MultiTaskVisualScoringModel(
        input_shape=(3, H, W), mask_shape=tuple(cfgm.mask_shape)
    )
    return model


def _load_vgg(cfgm, device):
    return load_vgg(freeze=cfgm.freeze, grad_layer=cfgm.grad_layer)


def _load_MFCN(cfgm, device):
    return MultiTaskVGG(mask_shape=tuple(cfgm.mask_shape), freeze=cfgm.freeze)


def _load_resnet(cfgm, device):
    return ResNetScoringModel(
        pretrained=True,
        input_channels=3,
        kernel_size=cfgm.kernel_size,
        num_classes=2,
        freeze=cfgm.freeze,
    )


def _load_vgg_segment(cfgm, device):
    H, W = cfgm.image_size
    model = SegmentVGG(
        seg_out_channels=1,
        mask_shape=tuple(cfgm.mask_shape),
        freeze=cfgm.freeze,
    )
    return model


# Registry: map your string names → loader functions
_MODEL_REGISTRY = {
    "cnn": _load_cnn,
    "deit_tiny": _load_deit_tiny,
    "multi_task": _load_multi_task,
    "vgg": _load_vgg,
    "MFCN": _load_MFCN,
    "resnet": _load_resnet,
    "vgg_seg": _load_vgg_segment,
}


def load_model(model_name: str, device: torch.device, cfgm) -> nn.Module:
    """
    Dispatch to the correct model‐builder, move it to device, then optionally load weights.
    cfgm : cfg.model
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name!r}")

    # 1) build
    model = _MODEL_REGISTRY[model_name](cfgm, device)

    # 2) to device
    model = model.to(device)

    # 3) optionally load pretrained weights
    if cfgm.load:
        if not os.path.exists(cfgm.weight_path):
            raise FileNotFoundError(f"Weight file not found: {cfgm.weight_path}")
        print(f"Loading weights from {cfgm.weight_path}")
        state = torch.load(cfgm.weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

    return model
