# config.py
from data.config_data import *
# --- Parameters ---
train_ratio = 0.8     # 80% train, 20% test
batch_size = 16
image_size = (224, 224)  # Resize all images to this size
mask_shape = (10, 10)  # Resize all masks to this size

model_name = "deit_tiny"  # Name of the model to load

# --- Paths ---
load_model = False  # Set to True to load the model from a path
weight_path = "checkpoints/deit_tiny_model.pth"  # Path to save the model weights

save_model = True  # Set to True to save the model after training





# --- Model Parameters ---
train = True  # Set to True to train the model
MODEL_PARAMS = {
    "learning_rate": 1e-4,
    "batch_size": batch_size,
    "epochs": 100,
}
LAMBDA_VISUAL = 0.5  # Poids du score visuel dans la fonction objective composite


# --- Heatmap Parameters ---
method = "grad_rollout"
heatmap_args = {
    "class_index":0 ,
    "target_layer": "model.conv3",
    "discard_ratio": 0.9  # Layer to visualize
        # Index of the class to visualize
    }