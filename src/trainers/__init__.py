from src.models.VisualScoringModel import train_model as _vis
from src.models.deit_tiny      import train_deit_no_precompute as _deit
from src.models.MultiTaskVisualModel import train_model_multi_task as _multi
from src.models.vgg            import train_vgg as _vgg
from src.models.MFCN           import train_model_multi_task as _mfcn
from src.models.resnet         import train_model as _resnet

TRAIN_REGISTRY = {
    "cnn": _vis,
    "deit_tiny":      _deit,
    "multi_task":     _multi,
    "vgg":            _vgg,
    "mfcn":           _mfcn,
    "resnet":         _resnet,
}

def get_trainer(name: str):
    try:
        return TRAIN_REGISTRY[name.lower()]
    except KeyError as e:
        raise ValueError(f"Unknown model name: {name}") from e