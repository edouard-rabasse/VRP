
import math, sys, os, torch, torchvision
import numpy as np
import torch.nn as nn

# import matplotlib.pyplot as plt





# load model

from torchvision import datasets, transforms
from src.data_loader import load_data

from models.VisualScoringModel import train_model

import cv2

from torch.utils.data import DataLoader


def show_mask_on_image(input_tensor, heatmap):
    # Resize to match input
    input_size = input_tensor.shape[2:]

    heatmap_resized = cv2.resize(heatmap, (84, 84))
    heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Overlay on input (grayscale to RGB)
    input_image = input_tensor[0, 0].cpu().numpy()
    input_image = cv2.cvtColor((input_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # resize to match input
    colored_heatmap = cv2.resize(colored_heatmap, (input_size[1], input_size[0]))

    print("Input Image shape:", input_image.shape)
    print("Heatmap shape:", colored_heatmap.shape)

    overlay = cv2.addWeighted(input_image, 0.5, colored_heatmap, 0.5, 0)
    return overlay

def get_heatmap(method, model, input_tensor, args):
    if method == 'gradcam':
        from models.VisualScoringModel import GradCAM
        target_layer = getattr(model, args['target_layer'])
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(input_tensor, class_index=args['class_index'])
    elif method == 'grad_rollout':
        from models.vit_explain.grad_rollout import VITAttentionGradRollout
        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args['discard_ratio'])
        heatmap = grad_rollout(input_tensor, category_index=args['class_index'])
    
    else:
        raise ValueError("Unknown method: {}".format(method))

    return heatmap

def load_model(model_name, device, cfg):

    if model_name == 'VisualScoringModel':
        from models.VisualScoringModel import VisualScoringModel
        image_size = cfg.image_size
        input_shape = (3, image_size[0], image_size[1])

        model = VisualScoringModel(input_shape=input_shape).to(device)

    elif model_name == 'deit_tiny':
        # from models.deit_tiny import load_deit
        # model = load_deit(model_name, device, out_features=2)
        # model.to(device)
        try:
            # model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
            import timm

            model = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=2)
        except RuntimeError as e:
            print(f"Error loading model from torch hub: {e}")
            print("Attempting to load model manually...")
            # Manual loading code here (if available)
            raise

# Changing the last layer to have 2 classes
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 2)
        model.to(device)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

    # Load the model weights
    if cfg.weight_path is not None:
        if os.path.exists(cfg.weight_path):
            print(f"Loading weights from {cfg.weight_path}")
            model.load_state_dict(torch.load(cfg.weight_path, map_location=device))
        else:
            raise FileNotFoundError(f"Weight file not found: {cfg.weight_path}")

    return model

def train(model_name,model, train_loader, test_loader, num_epochs, device, learning_rate):
    if model_name == 'VisualScoringModel':
        train_model(
            model,
            train_loader,
            test_loader,
            num_epochs=num_epochs,
            device=device,
            learning_rate=learning_rate,
            )
    elif model_name == 'deit_tiny':
        # Define your training loop for DEIT here
        from models.deit_tiny import precompute_deit_tiny_features, train_deit
        train_features = precompute_deit_tiny_features(model, train_loader, device=device)
        test_features = precompute_deit_tiny_features(model, test_loader, device=device)
        train_loader = DataLoader(train_features, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_features, batch_size=cfg.batch_size, shuffle=False)

        train_deit(model, train_loader, test_loader, device=device, num_epochs=num_epochs, learning_rate=learning_rate)

        


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
    data_path = cfg.DATA_DIR

    transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    ])

    model = load_model(
        model_name=cfg.model_name,
        device=device,
        cfg=cfg)

    train_loader, test_loader = load_data(
    data_path=data_path,
    batch_size=batch_size,
    transform=transform,
    train_ratio=train_ratio,
    image_size=image_size,
    num_workers=0,
    )

    

    if cfg.train:
        train(model_name=cfg.model_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=cfg.MODEL_PARAMS["epochs"],
            device=device,
            learning_rate=cfg.MODEL_PARAMS["learning_rate"],
        )


    # Ensure model is in eval mode
    model.eval()

    # test

    train_features, train_labels = next(iter(train_loader))
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




