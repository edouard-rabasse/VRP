# File: src/pipeline/config.py
from hydra import initialize, compose
from omegaconf import OmegaConf
import os
from pathlib import Path
import xml.etree.ElementTree as ET

# Base directory for the MSH solver
BASE_DIR = Path(__file__).parent.parent.parent / "MSH" / "MSH"

# Default Hydra overrides


# def get_cfg(overrides: list[str] | None = None):
#     """
#     Charge la configuration Hydra.
#     """
#     OmegaConf.register_new_resolver(
#         "env", lambda var_name: os.environ.get(var_name, "")
#     )
#     with initialize(version_base=None, config_path="../../config"):
#         return compose(config_name="config", overrides=overrides or DEFAULT_OVERRIDES)


def override_java_param(config_path: str, overrides: dict):
    """
    Override values in a Java-style .xml Properties file and preserve DOCTYPE.

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

    # Write to temporary string buffer first
    from io import BytesIO

    buffer = BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    xml_content = buffer.getvalue()

    # Manually insert DOCTYPE line after XML declaration
    xml_with_doctype = xml_content.replace(
        b"<?xml version='1.0' encoding='utf-8'?>",
        b"""<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">""",
    )

    # Write final content to file
    with open(config_path, "wb") as f:
        f.write(xml_with_doctype)


# Exemple d’usage
# override_java_param(
#     "MSH/MSH/configurationCustomCosts2.xml",
#     {"MAX_ITERATIONS": 1000, "USE_CONSTRAINTS": "true"},
# )

# # Ensuite tu lances Java
# # subprocess.run([...])
