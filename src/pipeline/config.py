# File: src/pipeline/config.py
from hydra import initialize, compose
from pathlib import Path
import xml.etree.ElementTree as ET

# Base directory for the MSH solver
BASE_DIR = Path(__file__).parent.parent.parent / "MSH" / "MSH"

# Default Hydra overrides
DEFAULT_OVERRIDES = [
    "data=config7",
    "model=resnet",
    "model.weight_path=checkpoints/resnet_8_30_7.pth",
    "model.load=true",
]


def get_cfg(overrides: list[str] | None = None):
    """
    Charge la configuration Hydra.
    """
    with initialize(version_base=None, config_path="../../config"):
        return compose(config_name="config", overrides=overrides or DEFAULT_OVERRIDES)


def override_java_param(config_path: str, overrides: dict):
    """
    Override values in a Java-style .xml Properties file.

    :param config_path: path to the configuration XML file
    :param overrides: dictionary of key-value pairs to override
    """
    tree = ET.parse(config_path)
    root = tree.getroot()

    for entry in root.findall("entry"):
        key = entry.get("key")
        if key in overrides:
            print(f"Overriding {key}: {entry.text} -> {overrides[key]}")
            entry.text = str(overrides[key])

    tree.write(config_path, encoding="utf-8", xml_declaration=True)


# Exemple d’usage
# override_java_param(
#     "MSH/MSH/configurationCustomCosts2.xml",
#     {"MAX_ITERATIONS": 1000, "USE_CONSTRAINTS": "true"},
# )

# # Ensuite tu lances Java
# # subprocess.run([...])
