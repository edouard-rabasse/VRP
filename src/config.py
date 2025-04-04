# config.py
DATA_DIR = "./data/test/"

# --- Parameters ---
train_ratio = 0.8     # 80% train, 20% test
batch_size = 32
image_size = (284, 284)  # Resize all images to this size


RAW_DATA_DIR = DATA_DIR + "raw/"
PROCESSED_DATA_DIR = DATA_DIR + "processed/"
MODEL_PARAMS = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
}
LAMBDA_VISUAL = 0.5  # Poids du score visuel dans la fonction objective composite
