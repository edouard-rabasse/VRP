import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
from src.models import load_model
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.load_model = True
    model = load_model(
        model_name="vgg", device=device, cfgm=cfg.model
    )  # Adjust model name as needed
    print(model)
    # Choose the layer and channel to maximize
    target_layer = model.features[28]
    target_channel = 0  # try from 0 to 511

    # Hook to capture activation
    activation = None

    def hook_fn(module, input, output):
        global activation
        activation = output

    hook = target_layer.register_forward_hook(hook_fn)

    # Start from a random image
    input_img = torch.randn(1, 3, 224, 224, requires_grad=True, device=device)

    # Normalize like ImageNet
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Optimization loop
    optimizer = torch.optim.Adam([input_img], lr=0.05)
    for i in range(100):
        optimizer.zero_grad()
        model(normalizer(input_img.squeeze(0)).unsqueeze(0))

        loss = -activation[0, target_channel].mean()  # maximize activation
        loss.backward()
        optimizer.step()

        # Optional: clamp image to valid range
        with torch.no_grad():
            input_img.clamp_(0, 1)

    # Detach and visualize
    img = input_img.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(f"Maximizing layer[28] channel[{target_channel}]")
    plt.axis("off")
    plt.savefig("max_activation.png", bbox_inches="tight")

    hook.remove()


if __name__ == "__main__":
    main()
