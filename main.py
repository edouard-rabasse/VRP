
import math, sys, os, torch, torchvision
import numpy as np
import torch.nn as nn

# import matplotlib.pyplot as plt





# load model

from torchvision import datasets, transforms
from src.data_loader_mask import load_data_mask

from models.VisualScoringModel import train_model

import cv2

from torch.utils.data import DataLoader
from src.visualization import get_heatmap, show_mask_on_image
from src.load_model import load_model
from src.train import train








if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "src/config.py"
    sys.path.append(os.path.dirname(cfg))
    cfg = __import__(os.path.basename(cfg).replace('.py', ''))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    print("CUDA version:", torch.version.cuda)

    image_size = cfg.image_size
    batch_size = cfg.batch_size
    train_ratio = cfg.train_ratio
    original_path = cfg.original_path
    modified_path = cfg.modified_path
    mask_path = cfg.mask_path

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    ])

    model = load_model(
        model_name=cfg.model_name,
        device=device,
        cfg=cfg)

    train_loader, test_loader = load_data_mask(original_path, modified_path, batch_size=32, transform=transform, train_ratio=0.8, 
              image_size=(224, 224), num_workers=4, mask_path=None)

    

    if cfg.train:
        train(model_name=cfg.model_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=cfg.MODEL_PARAMS["epochs"],
            device=device,
            learning_rate=cfg.MODEL_PARAMS["learning_rate"],
            cfg=cfg
        )


    # Ensure model is in eval mode
    model.eval()

    # test

    train_features, train_labels, train_mask = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]

    img = img.permute(1, 2, 0).cpu().numpy()

    input_tensor = train_features[0].unsqueeze(0).to(device)  # Add batch dimension

    # Generate heatmap for class 2
  
    heatmap = get_heatmap(cfg.method, model, input_tensor, cfg.heatmap_args)
    print(type(heatmap))
    print(heatmap.shape)

    overlay = show_mask_on_image(input_tensor, heatmap)

    

    # Save or show the result
    cv2.imwrite("gradcam_overlay.png", overlay)




