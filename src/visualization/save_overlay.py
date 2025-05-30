import os
from PIL import Image


def save_overlay(overlay, output_dir, fname):
    os.makedirs(output_dir, exist_ok=True)
    out_p = os.path.join(output_dir, fname)
    Image.fromarray(overlay).save(out_p)
