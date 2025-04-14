
import math, sys, os, torch, torchvision
import numpy as np
import torch.nn as nn

# import matplotlib.pyplot as plt





# load model

from torchvision import transforms
from src.data_loader_mask import load_data_mask


import cv2
from src.visualization import get_heatmap, show_mask_on_image
from src.load_model import load_model









if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "src/config.py"
    sys.path.append(os.path.dirname(cfg))
    cfg = __import__(os.path.basename(cfg).replace('.py', ''))




    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

    image_size = cfg.image_size
    batch_size = cfg.batch_size
    train_ratio = cfg.train_ratio
    original_path = cfg.original_path
    modified_path = cfg.modified_path
    mask_path = cfg.mask_path

    ## Define the transform to resize the images and convert them to tensors
    ## TODO : use an external transform function

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # Normalize with ImageNet's mean and std if using a pretrained model like VGG16
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Transform for testing (deterministic)
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    print("Loading model...")
    model = load_model(
        model_name=cfg.model_name,
        device=device,
        cfg=cfg)
    print(f"Model {cfg.model_name} loaded.")

    print("Loading data...")
    train_loader, test_loader = load_data_mask(
        original_path,
        modified_path,
        batch_size=32,
        transform_train=transform_train,
        transform_test=transform_test,
        train_ratio=0.8,
        image_size=(224, 224),
        num_workers=4,
        mask_path=mask_path,
        num_max=None
        )
    print("Data loaded.")

    

    if cfg.train:
        from src.train import train
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

    model.to(device)
    # extract 1 image from the modified path
    # print(vars(test_loader.dataset.dataset))
    # for i in range(len(test_loader.dataset.dataset)):
    #     print(test_loader.dataset.dataset.modified_images[i])
    #     if test_loader.dataset.dataset.modified_images[i][1] == 1:
    #         index = i
    #         break
   

    modified_path = "MSH/MSH/plots/configuration5/Plot_5.png"

    
    modified_img = cv2.imread(modified_path)
    modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB)
    modified_tensor = transform(modified_img).unsqueeze(0).to(device)

    cv2.imwrite("output/modified.png", modified_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
    heatmap= get_heatmap(cfg.method, model, modified_tensor, cfg.heatmap_args)
    print(heatmap.shape)

    overlay = show_mask_on_image(modified_tensor, heatmap, alpha=0.5)
    cv2.imwrite(f"output/{cfg.method}.png", overlay)