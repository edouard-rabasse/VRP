
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
from src.transform import image_transform_train, image_transform_test, mask_transform, denormalize








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
    

    print("Loading model...")
    model = load_model(
        model_name=cfg.model_name,
        device=device,
        cfg=cfg)
    print(f"Model {cfg.model_name} loaded.")

    print(type(image_transform_train(size=image_size)))

    print("Loading data...")
    train_loader, test_loader = load_data_mask(
        original_path=original_path,
        modified_path=modified_path,
        batch_size=32,
        image_transform_train=image_transform_train(size=image_size),
        image_transform_test=image_transform_test(size=image_size),
        mask_transform_train=mask_transform(size=image_size),
        mask_transform_test=mask_transform(size=image_size),
        train_ratio=train_ratio,
        image_size=image_size,
        num_workers=2,
        mask_path=mask_path
    )
    print("Data loaded.")

    

    if cfg.train:
        from src.train_functions import train
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
   
    
    plot_numbers = range(1, 15)  # Adjust the range as needed
    for plot_number in plot_numbers:
        img_path = f"{modified_path}Plot_{plot_number}.png"
        



        modified_img = cv2.imread(img_path)
        # modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB)
        modified_tensor = image_transform_test(image_size)(modified_img).unsqueeze(0)
        # with torch.no_grad():
        #     modified_tensor = modified_tensor.to(device)
        #     pred = model(modified_tensor)
        # pred = torch.argmax(pred, dim=1).item()
        # print(f"Prediction: {pred}")

        # cv2.imwrite("output/modified.png", modified_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
        heatmap= get_heatmap(cfg.method, model, modified_tensor, cfg.heatmap_args)
        

        tensor = denormalize(modified_tensor.squeeze(0).cpu())

        overlay = show_mask_on_image(tensor, heatmap, alpha=0.5)
        cv2.imwrite(f"output/{cfg.method}_{plot_number}.png", overlay)
        del modified_img, modified_tensor
        del heatmap, tensor, overlay
        torch.cuda.empty_cache()