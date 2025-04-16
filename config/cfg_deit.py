# config.py
original_path = "MSH/MSH/plots/configuration1/"
modified_path = "MSH/MSH/plots/configuration4/"
mask_path = "data/MSH/mask/"

# --- Parameters ---
train_ratio = 0.8     # 80% train, 20% test
batch_size = 16
image_size = (224, 224)  # Resize all images to this size

model_name = "deit_tiny"  # Name of the model to load

# --- Paths ---
weight_path = None






# --- Model Parameters ---
train = True  # Set to True to train the model
MODEL_PARAMS = {
    "learning_rate": 0.001,
    "batch_size": batch_size,
    "epochs": 50,
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