import torch


def recursive_getattr(model: torch.nn.Module, attr_path: str) -> torch.nn.Module:
    """
    Recursively get a nested attribute from a PyTorch module using dot notation.

    This function allows accessing deeply nested layers in a model using a string path.
    It handles both attribute access (using getattr) and indexing (for Sequential modules).

    Args:
        model: The root PyTorch module to start the attribute search from.
        attr_path: Dot-separated string path to the desired attribute/layer.
                  Can include numeric indices for Sequential containers.
                  Example: "features.0.conv1" or "layer1.0.bn1"

    Returns:
        The target PyTorch module/layer found at the specified path.

    Example:
        >>> model = torchvision.models.resnet18()
        >>> layer = recursive_getattr(model, "layer1.0.conv1")
        >>> # Equivalent to: model.layer1[0].conv1
    """
    attrs = attr_path.split(".")
    for attr in attrs:
        if attr.isdigit():
            model = model[int(attr)]
        else:
            model = getattr(model, attr)
    return model
