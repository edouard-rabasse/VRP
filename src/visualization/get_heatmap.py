import numpy as np
import torch
from .get_attr import recursive_getattr


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

        target_layer = recursive_getattr(model, args.target_layer)
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(input_tensor, class_index=args.class_index)
    elif method == "grad_rollout":
        from src.models.vit_explain.grad_rollout import VITAttentionGradRollout

        grad_rollout = VITAttentionGradRollout(
            model, discard_ratio=args.discard_ratio, device=device
        )

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

            # heatmap = cv2.resize(
            #     heatmap, (input_tensor.shape[2], input_tensor.shape[3])
            # )
            # heatmap = heatmap* 255  # Scale to [0, 255]

    elif method == "grad_cam_vgg":
        from src.models.VisualScoringModel import GradCAM

        target_layer = model.features[29]  # Assuming the last layer is the target layer
        # target_layer = model.block5.conv3  # Assuming the last layer is the target layer

        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam(input_tensor, class_index=args["class_index"])
    elif method == "seg":
        with torch.no_grad():
            model.eval()
            seg_logits = model(input_tensor)
            heatmap = seg_logits[0, 0]
            # Assuming the first channel is the one of interest
            # apply softmax to the heatmap
            heatmap = torch.nn.functional.softmax(heatmap, dim=0).cpu().numpy()

    else:
        raise ValueError("Unknown method: {}".format(method))

    thresh = np.percentile(heatmap, 95)  # Threshold for heatmap
    # print("Threshold for heatmap:", thresh)
    heatmap = np.clip(heatmap, thresh, 1)  # Clip values to [thresh,1]
    # normalize the heatmap to [0, 1]
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)

    return heatmap
