import cv2
import numpy as np
import torch


def resize_heatmap(heatmap, target_size, interpolation=cv2.INTER_LINEAR):
    """Resize the heatmap to match the target size."""
    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=interpolation)
    heatmap_resized = (heatmap_resized - np.min(heatmap_resized)) / (
        np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-8
    )

    return heatmap_resized


def show_mask_on_image(input, heatmap, alpha=0.5, interpolation=cv2.INTER_LINEAR):
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
    input_image = (input_image * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
    # print("maximum over the whle input:", np.max(input_image))

    # print("Input Image shape:", input_image.shape)
    # print("Heatmap shape:", colored_heatmap.shape)

    overlay = cv2.addWeighted(input_image, alpha, colored_heatmap_rgb, 1 - alpha, 0)
    return overlay


def get_heatmap(method, model, input_tensor, args, device="cpu"):
    """
    Generate a heatmap using the specified method.
    ## Args:
    -  method (str): The method to use for generating the heatmap.
    - model (torch.nn.Module): The model to use for generating the heatmap.
    - input_tensor (torch.Tensor): The input tensor to the model.
    -args (dict): Additional arguments for the method.
    ## Returns:
    - heatmap (numpy.ndarray): The generated heatmap."""
    input_tensor = input_tensor.to(device)

    if method == "gradcam":
        from src.models.VisualScoringModel import GradCAM

        target_layer = getattr(model, args.target_layer)
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(input_tensor, class_index=args.class_index)
    elif method == "grad_rollout":
        from src.models.vit_explain.grad_rollout import VITAttentionGradRollout

        grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)

        heatmap = grad_rollout(input_tensor, category_index=args.class_index)
        # print("Grad Rollout heatmap shape:", heatmap.shape)

    elif method == "multi_task":
        with torch.no_grad():
            model.eval()
            cls_logits, seg_logits = model(input_tensor)
            heatmap = seg_logits[
                0, 0
            ]  # Assuming the first channel is the one of interest
            # apply softmax to the heatmap
            heatmap = torch.nn.functional.softmax(heatmap, dim=0).cpu().numpy()

            heatmap = cv2.resize(
                heatmap, (input_tensor.shape[2], input_tensor.shape[3])
            )
            # heatmap = heatmap* 255  # Scale to [0, 255]

    elif method == "grad_cam_vgg":
        from src.models.VisualScoringModel import GradCAM

        target_layer = model.features[29]  # Assuming the last layer is the target layer
        # target_layer = model.block5.conv3  # Assuming the last layer is the target layer

        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(input_tensor, class_index=args["class_index"])

    else:
        raise ValueError("Unknown method: {}".format(method))

    thresh = np.percentile(heatmap, 95)  # Threshold for heatmap
    # print("Threshold for heatmap:", thresh)
    heatmap = np.clip(heatmap, thresh, 1)  # Clip values to [thresh,1]
    # normalize the heatmap to [0, 1]
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

    return heatmap


def get_mask(original_image, modified_image):
    """
    Compute the mask by subtracting the modified image from the original image.
    """
    mask = cv2.absdiff(original_image, modified_image)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # Binarize the mask
    return mask


def intersection_with_heatmap(heatmap, mask):
    """
    Compute the intersection of the heatmap and the mask.
    """
    # Ensure heatmap and mask are in the same format
    if len(heatmap.shape) == 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    # Resize heatmap to match mask size
    heatmap_resized = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]))

    # Threshold the heatmap to create a binary mask
    _, heatmap_binary = cv2.threshold(
        heatmap_resized, 0.5 * 255, 255, cv2.THRESH_BINARY
    )

    # Compute intersection
    intersection = cv2.bitwise_and(heatmap_binary, mask)

    return intersection
