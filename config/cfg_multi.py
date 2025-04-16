# config.py
original_path = "MSH/MSH/plots/configuration1/"
modified_path = "MSH/MSH/plots/configuration3/"
mask_path = "data/MSH/mask/"
# --- Parameters ---
train_ratio = 0.8     # 80% train, 20% test
batch_size = 8
image_size = (284, 284)  # Resize all images to this size

model_name = "multi_task"  # Name of the model to load

# --- Paths ---
weight_path = None






# --- Model Parameters ---
train = False  # Set to True to train the model
MODEL_PARAMS = {
    "learning_rate": 0.0001,
    "batch_size": batch_size,
    "epochs": 20,
    "lambda_seg" : 1.0
}
LAMBDA_VISUAL = 0.5  # Poids du score visuel dans la fonction objective composite


# --- Heatmap Parameters ---
method = "multi_task"
heatmap_args = {
    "class_index":0 ,
    "target_layer": "conv3",  # Index of the class to visualize
    }