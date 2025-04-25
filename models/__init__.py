# src/models/__init__.py
import os
import timm
import torch
import torch.nn as nn

from .VisualScoringModel       import VisualScoringModel
from .MultiTaskVisualModel     import MultiTaskVisualScoringModel
from .vgg                      import load_vgg
from .MFCN                     import MultiTaskVGG

def _load_cnn(cfg, device):
    H, W = cfg.image_size
    model = VisualScoringModel(input_shape=(3, H, W))
    return model

def _load_deit_tiny(cfg, device):
    # timm already handles head replacement when you pass num_classes
    return timm.create_model(
        'deit_tiny_patch16_224',
        pretrained=True,
        num_classes=2
    )

def _load_multi_task(cfg, device):
    H, W = cfg.image_size
    model = MultiTaskVisualScoringModel(
        input_shape=(3, H, W),
        mask_shape=cfg.mask_shape
    )
    return model

def _load_vgg(cfg, device):
    return load_vgg()

def _load_MFCN(cfg, device):
    return MultiTaskVGG(mask_shape=cfg.mask_shape)

# Registry: map your string names → loader functions
_MODEL_REGISTRY = {
    'cnn'         : _load_cnn,
    'deit_tiny'   : _load_deit_tiny,
    'multi_task'  : _load_multi_task,
    'vgg'         : _load_vgg,
    'MFCN'        : _load_MFCN,
}

def load_model(model_name: str, device: torch.device, cfg) -> nn.Module:
    """
    Dispatch to the correct model‐builder, move it to device, then optionally load weights.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name!r}")

    # 1) build
    model = _MODEL_REGISTRY[model_name](cfg, device)

    # 2) to device
    model = model.to(device)

    # 3) optionally load pretrained weights
    if cfg.load_model:
        if not os.path.exists(cfg.weight_path):
            raise FileNotFoundError(f"Weight file not found: {cfg.weight_path}")
        print(f"Loading weights from {cfg.weight_path}")
        state = torch.load(cfg.weight_path, map_location=device)
        model.load_state_dict(state)

    return model
