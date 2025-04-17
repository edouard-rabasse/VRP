# config.py
# config.py
from .data.config_data import *
# --- Parameters ---
train_ratio = 0.8     # 80% train, 20% test
batch_size = 16
image_size = (224, 224)  # Resize all images to this size

model_name = "vgg"  # Name of the model to load

# --- Paths ---
weight_path = None






# --- Model Parameters ---
train = True  # Set to True to train the model
MODEL_PARAMS = {
    "learning_rate": 0.0001,
    "batch_size": batch_size,
    "epochs": 50,
}
LAMBDA_VISUAL = 0.5  # Poids du score visuel dans la fonction objective composite


# --- Heatmap Parameters ---
method = "grad_cam_vgg"
heatmap_args = {
    "class_index":0 ,
    "target_layer": "features[-1]",  # Index of the class to visualize
    }