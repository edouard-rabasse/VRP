
from data.config_data import *
# --- Parameters ---
train_ratio = 0.8     # 80% train, 20% test
batch_size = 8
image_size = (224, 224)  # Resize all images to this size
mask_shape = (10, 10)  # Resize all masks to this size

model_name = "cnn"  # Name of the model to load

# --- Paths ---
load_model = True 
weight_path = "checkpoints/CNNModel_model.pth"  # Path to save the model weights
save_model = False






# --- Model Parameters ---
train = False  # Set to True to train the model
MODEL_PARAMS = {
    "learning_rate": 0.0001,
    "batch_size": batch_size,
    "epochs": 20,
}
LAMBDA_VISUAL = 0.5  # Poids du score visuel dans la fonction objective composite


# --- Heatmap Parameters ---
method = "gradcam"
heatmap_args = {
    "class_index":0 ,
    "target_layer": "conv3",  # Index of the class to visualize
    }
