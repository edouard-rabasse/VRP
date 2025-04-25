
import math, sys, os, torch, torchvision
import numpy as np
import torch.nn as nn

# import matplotlib.pyplot as plt





# load model

from torchvision import transforms
from src.data_loader_mask import load_data_train_test


from PIL import Image
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
    train_original_path = cfg.train_original_path
    test_original_path = cfg.test_original_path
    train_modified_path = cfg.train_modified_path
    test_modified_path = cfg.test_modified_path
    train_mask_path = cfg.train_mask_path
    test_mask_path = cfg.test_mask_path

    ## Define the transform to resize the images and convert them to tensors

    print("Loading model...")

    
    model = load_model(
        model_name=cfg.model_name,
        device=device,
        cfg=cfg)
    print(f"Model {cfg.model_name} loaded.")


    print("Loading data...")
    train_loader, test_loader = load_data_train_test(
        train_original_path=train_original_path,
        test_original_path=test_original_path,
        train_modified_path=train_modified_path,
        test_modified_path=test_modified_path,
        mask_path_train=train_mask_path,
        mask_path_test=test_mask_path,
        batch_size=batch_size,
        image_transform_train=image_transform_train(size=image_size),
        image_transform_test=image_transform_test(size=image_size),
        mask_transform_train=mask_transform(size=cfg.mask_shape),
        mask_transform_test=mask_transform(size=cfg.mask_shape),
        image_size=image_size,
        num_workers=0
    )
    print("Data loaded.")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

    if cfg.train:
        from src.train_functions import train
        results = train(model_name=cfg.model_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=cfg.MODEL_PARAMS["epochs"],
            device=device,
            learning_rate=cfg.MODEL_PARAMS["learning_rate"],
            cfg=cfg
        )
        # save results to a file
        from src.evaluation import get_confusion_matrix
        matrix = get_confusion_matrix(model, test_loader, device=device)
        results.append(f"Confusion matrix: \n {matrix}")

        
        try:
            f = open(f"results/{cfg.model_name}_{cfg.cfg_number}.txt", "x") 
        except FileExistsError:
            f = open(f"results/{cfg.model_name}_{cfg.cfg_number}.txt", "w")
        f.writelines('\n'.join(results))
    if cfg.save_model:
        print(f"Saving model to {cfg.weight_path}...")
        from src.load_model import save_model
        save_model(model, cfg.weight_path)
        print("Model saved.")


    # Ensure model is in eval mode
    model.eval()
    model.to(device)
    # extract 1 image from the modified path
    # print(vars(test_loader.dataset.dataset))
    # for i in range(len(test_loader.dataset.dataset)):
    #     print(test_loader.dataset.dataset.modified_images[i])
    #     if test_loader.dataset.dataset.modified_images[i][1] == 1:
    #         index = i
    #         break7

    
    test_adresses = os.listdir(test_modified_path)
    test_adresses = [x for x in test_adresses if x.endswith('.png')]

    for adress in test_adresses:
        # break
        img_path = f"{test_original_path}{adress}"
        



        modified_img = Image.open(img_path).convert("RGB")
        # modified_img = cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB)
        modified_tensor = image_transform_test(image_size)(modified_img).unsqueeze(0)
        # cls, seg = model(modified_tensor.to(device))
        # print(seg.shape)

        # with torch.no_grad():
        #     modified_tensor = modified_tensor.to(device)
        #     pred = model(modified_tensor)
        # proba = torch.nn.Softmax(dim=1)(pred)
        # pred = torch.argmax(pred, dim=1).item()
        # print(f"Prediction: {pred}")
        # print("proba", proba)

        # cv2.imwrite("output/modified.png", modified_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
        heatmap= get_heatmap(cfg.method, model, modified_tensor, cfg.heatmap_args, device=device)
        

        tensor = denormalize(modified_tensor.squeeze(0).cpu())

        mask_pth = f"{test_mask_path}{adress}"
        mask = Image.open(mask_pth).convert("L")
        # resize mask to the same size as the image
        mask = torchvision.transforms.functional.resize(mask, (tensor.shape[2], tensor.shape[1]), interpolation=transforms.InterpolationMode.NEAREST)
        # mask = cv2.resize(mask, (tensor.shape[2], tensor.shape[1]), interpolation=cv2.INTER_LINEAR)
        mask = mask_transform(size=image_size)(mask)



        overlay = show_mask_on_image(mask, heatmap, alpha=0.5)

        # write the overlay to disk
        from torchvision.utils import save_image
        # save_image(overlay, f"output/{cfg.method}/{adress}", normalize=True, scale_each=True)
        # create the directory if it doesn't exist
        if not os.path.exists(f"output/{cfg.method}_{cfg.model_name}"):
            os.makedirs(f"output/{cfg.method}_{cfg.model_name}")
        cv2.imwrite(f"output/{cfg.method}_{cfg.model_name}/{adress}", overlay)
        # print("saved")
        del modified_img, modified_tensor
        del heatmap, tensor, overlay
        torch.cuda.empty_cache()