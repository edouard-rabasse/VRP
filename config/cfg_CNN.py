# config.py
original_path = "MSH/MSH/plots/configuration1/"
modified_path = "MSH/MSH/plots/configuration5/"
mask_path = "data/MSH/mask/"
# --- Parameters ---
train_ratio = 0.8     # 80% train, 20% test
batch_size = 8
image_size = (224, 224)  # Resize all images to this size

model_name = "VisualScoringModel"  # Name of the model to load

# --- Paths ---
weight_path = None






# --- Model Parameters ---
train = True  # Set to True to train the model
MODEL_PARAMS = {
    "learning_rate": 0.001,
    "batch_size": batch_size,
    "epochs": 10,
}
LAMBDA_VISUAL = 0.5  # Poids du score visuel dans la fonction objective composite


# --- Heatmap Parameters ---
method = "gradcam"
heatmap_args = {
    "class_index":0 ,
    "target_layer": "conv3",  # Index of the class to visualize
    }