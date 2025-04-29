import yaml
from types import SimpleNamespace
from omegaconf import DictConfig, OmegaConf

def save_model(model, path):
    # Sauvegarde du modèle entraîné
    pass

def load_cfg_yaml(path: str):
    """Load a YAML file and return a dot-accessible namespace."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    def to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [to_ns(x) for x in obj]
        return obj

    return to_ns(raw)



def load_cfg(path: str) -> SimpleNamespace:
    """Lit un YAML OmegaConf et renvoie un objet dot-accessible."""
    cfg: DictConfig = OmegaConf.load(path)

    # # ─ Optionnel : conversion en SimpleNamespace (utile pour ton code existant)
    # def to_ns(obj):
    #     if isinstance(obj, DictConfig):
    #         return SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
    #     return obj

    # return to_ns(cfg)
    return cfg