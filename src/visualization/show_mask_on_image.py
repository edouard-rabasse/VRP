import cv2
import numpy as np
from .resize_heatmap import resize_heatmap
import torch


def show_mask_on_image(
    input: torch.Tensor, heatmap: np.ndarray, alpha=0.5, interpolation=cv2.INTER_NEAREST
):
    """
    Overlay the heatmap on the input image.
    ## Args:
    - input_tensor (torch.Tensor): The input tensor to the model, dimension (N, C, H, W) or (C, H, W), in RGB format.
    - heatmap (numpy.ndarray): The heatmap to overlay.
    - alpha (float): The transparency level for the overlay. Higher values mean more of the input image is visible.
    ## Returns:
    - overlay (numpy.ndarray): The overlayed image.
    """
    # Resize to match input
    if len(input.shape) == 4:
        input = input.squeeze(0)
    if type(input) == torch.Tensor:
        input = input.cpu().numpy()

    # check that it is in the right format,
    if len(input.shape) != 3:
        raise ValueError(
            "[show_mask_on_image]Input tensor must have 3 channels (C, H, W) format."
        )
    if input.shape[0] == 1:
        input = np.repeat(input, 3, axis=0)

    # print("input_shape:", input.shape)
    input_size = (input.shape[1], input.shape[2])

    heatmap_resized = resize_heatmap(
        heatmap, (input_size[0], input_size[1]), interpolation=interpolation
    )
    heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
    # print("max over heatmap:", np.max(heatmap_resized))
    # print("max of input:", np.max(input))
    colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Overlay on input (grayscale to RGB)
    input_image = input.transpose(1, 2, 0)
    print("min/max input before scaling:", input.min(), input.max())
    input_image = (input_image * 255).astype(np.uint8)  # Convert to uint8 for OpenCV

    # print("maximum over the whle input:", np.max(input_image))

    # print("Input Image shape:", input_image.shape)
    # print("Heatmap shape:", colored_heatmap.shape)

    overlay = cv2.addWeighted(input_image, alpha, colored_heatmap_rgb, 1 - alpha, 0)
    return overlay
