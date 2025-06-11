import torch
import pytest
from types import SimpleNamespace
from src.pipeline.model_loader import ModelLoader


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.warmup_called = False
        self.last_input = None

    def forward(self, x):
        # record that forward was called for warm-up
        self.warmup_called = True
        # store the input tensor for inspection
        self.last_input = x
        # return a dummy tensor
        return torch.zeros(1)


@pytest.fixture
def dummy_cfg():
    # simple config object with a name attribute
    return SimpleNamespace(name="dummy_model")


def test_model_loader_stores_config_and_device(dummy_cfg):
    loader = ModelLoader(dummy_cfg, device="cpu")
    assert loader.model_cfg is dummy_cfg
    assert loader.device == "cpu"


def test_load_invokes_load_model_and_warmup(monkeypatch, dummy_cfg):
    # Prepare a dummy model instance
    dummy_model = DummyModel()

    # Stub load_model to return our dummy_model
    called = {}

    def fake_load_model(name, device, cfgm):
        called["args"] = (name, device, cfgm)
        return dummy_model

    monkeypatch.setattr("src.pipeline.model_loader.load_model", fake_load_model)

    # Create loader and call load()
    loader = ModelLoader(dummy_cfg, device="cpu")
    returned_model = loader.load()

    # Check that load_model was called with correct arguments
    assert called["args"] == ("dummy_model", "cpu", dummy_cfg)

    # The returned model should be the same dummy_model
    assert returned_model is dummy_model

    # After load(), the dummy_model.forward should have been called with a zero tensor of shape [1,3,224,224]
    assert dummy_model.warmup_called, "Model warm-up forward was not called"
    inp = dummy_model.last_input
    assert isinstance(inp, torch.Tensor)
    assert inp.shape == (1, 3, 224, 224)
    assert inp.device.type == ("cpu")


def test_load_on_cuda_passes_device_correctly(monkeypatch, dummy_cfg):
    # simulate a cuda device
    device_str = "cuda"
    dummy_model = DummyModel()

    def fake_load_model(name, device, cfgm):
        called_device = device
        # wrap dummy model to check parameter device tag
        dummy_model.to(device)
        return dummy_model

    monkeypatch.setattr("src.pipeline.model_loader.load_model", fake_load_model)

    loader = ModelLoader(dummy_cfg, device=device_str)
    returned_model = loader.load()

    # The model should now have parameters on the cuda device
    # We check one parameter's device
    for param in returned_model.parameters():
        assert param.device.type == device_str
        break

    # Also warm-up forward should be called on the correct device
    assert returned_model.warmup_called
    inp = returned_model.last_input
    assert inp.device.type == device_str
