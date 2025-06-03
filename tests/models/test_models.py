import pytest
import torch
from src.models import load_model


@pytest.mark.parametrize(
    "model_name",
    [
        "cnn",
        "deit_tiny",
        "multi_task",
        "vgg",
        "MFCN",
        "resnet",
        "vgg_seg",
    ],
)
def test_load_all_models(model_name):
    class DummyCfg:
        load = False
        freeze = False
        grad_layer = 5
        image_size = (224, 224)
        mask_shape = (10, 10)

        if model_name == "resnet":
            # ResNet expects kernel_size as int (e.g., 3)
            kernel_size = 3
        else:
            # Other models expect kernel_size as string "3,3"
            kernel_size = "3,3"

    device = torch.device("cpu")

    model = load_model(model_name, device, DummyCfg())

    assert isinstance(model, torch.nn.Module)

    # Optionnel : forward dummy input to check no error
    # Note : adaptez la taille du batch et input selon modèle
    dummy_input = torch.randn(1, 3, 224, 224)
    if hasattr(model, "forward"):
        _ = model(dummy_input)
    elif hasattr(model, "forward_features"):
        _ = model.forward_features(dummy_input)
    else:
        # Pour certains modèles, adapte ici si besoin
        pass
