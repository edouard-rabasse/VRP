import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from src.models import load_model


def test_load_model_valid(monkeypatch):
    # Mock torch.load pour éviter vrai chargement
    def mock_torch_load(path, **kwargs):
        return {"state_dict": "fake_state"}

    monkeypatch.setattr(torch, "load", mock_torch_load)

    # Mock os.path.exists pour forcer à True l’existence du fichier de poids
    monkeypatch.setattr(os.path, "exists", lambda path: True)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def to(self, device):
            self.called = True
            return self

        def load_state_dict(self, state):
            self.loaded_state = state

    monkeypatch.setattr(
        "src.models._MODEL_REGISTRY", {"dummy": lambda cfg, device: DummyModel()}
    )

    class Cfg:
        load = True
        weight_path = "fake_path"

    model = load_model("dummy", torch.device("cpu"), Cfg())

    assert isinstance(model, torch.nn.Module)
    assert model.called
    assert hasattr(model, "loaded_state")
    assert model.loaded_state == {"state_dict": "fake_state"}


def test_load_model_unknown_raises():
    class DummyCfg:
        load = False
        weight_path = ""

    with pytest.raises(ValueError, match="Unknown model name"):
        load_model("unknown_model_name", torch.device("cpu"), DummyCfg())
