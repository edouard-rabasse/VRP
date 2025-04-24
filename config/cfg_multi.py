# config.py
from data.config_data import *
# --- Parameters ---
train_ratio = 0.8     # 80% train, 20% test
batch_size = 8
image_size = (284, 284)  # Resize all images to this size
mask_shape = (10, 10)  # Resize all masks to this size

model_name = "multi_task"  # Name of the model to load

# --- Paths ---
load_model = False 
weight_path = "checkpoints/multi_task_model.pth"  # Path to save the model weights
save_model = True





# --- Model Parameters ---
train = True  # Set to True to train the model
MODEL_PARAMS = {
    "learning_rate": 5e-4,
    "batch_size": batch_size,
    "epochs": 20,
    "lambda_seg" : 0.2
}
LAMBDA_VISUAL = 0.5  # Poids du score visuel dans la fonction objective composite


# --- Heatmap Parameters ---
method = "multi_task"
heatmap_args = {
    "class_index":0 ,
    "target_layer": "conv3",  # Index of the class to visualize
    }