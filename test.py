
import math, sys, os, torch, torchvision
import numpy as np

# import matplotlib.pyplot as plt





# load model
# from src.data_loader import load_data

# from models.VisualScoringModel import train_model

import cv2

from torch.utils.data import DataLoader





def load_model(model_name, device, cfg):

    if model_name == 'deit_tiny':
        # from models.deit_tiny import load_deit
        print("loading deit_tiny")
        # model = load_deit(model_name, device, out_features=2)
        # model.to(device)
        print("Torch Hub cache dir:", torch.hub.get_dir())
        try:
            model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        except RuntimeError as e:
            print(f"Error loading model from torch hub: {e}")
            print("Attempting to load model manually...")
            # Manual loading code here (if available)
            raise

# Changing the last layer to have 2 classes
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 2)
    else:
        raise ValueError("Unknown model name: {}".format(model_name))


    return model.to(device)




def main():
    # cfg = sys.argv[1] if len(sys.argv) > 1 else "src/config.py"
    # sys.path.append(os.path.dirname(cfg))
    # cfg = __import__(os.path.basename(cfg).replace('.py', ''))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    print("CUDA version:", torch.version.cuda)


    model = load_model(
        model_name="deit_tiny",
        device=device,
        cfg=None)

    

    # Ensure model is in eval mode
    model.eval()

    # test
if __name__ == "__main__":
    main()



