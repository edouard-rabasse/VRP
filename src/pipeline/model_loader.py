# File: src/pipeline/model_loader.py
import torch
from src.models import load_model


class ModelLoader:
    """
    Charge et prépare le modèle PyTorch pour l'inférence.
    """

    def __init__(self, model_cfg, device: str):
        self.model_cfg = model_cfg
        self.device = device

    def load(self) -> torch.nn.Module:
        model = load_model(
            self.model_cfg.name,
            self.device,
            cfgm=self.model_cfg,
        )
        # Warm-up
        dummy = torch.zeros(1, 3, 224, 224, device=self.device)
        model(dummy)
        return model
