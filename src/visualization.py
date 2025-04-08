import cv2
import numpy as np


def resize_heatmap(heatmap, target_size):
    """Resize the heatmap to match the target size."""
    heatmap_resized = cv2.resize(heatmap, target_size)
    heatmap_resized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))
    
    colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    return colored_heatmap

def show_mask_on_image(input_tensor, heatmap):
    # Resize to match input
    input_size = input_tensor.shape[2:]

    heatmap_resized = resize_heatmap(heatmap, (input_size[1], input_size[0]))
    heatmap_resized = (heatmap_resized * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Overlay on input (grayscale to RGB)
    input_image = input_tensor[0, 0].cpu().numpy()
    input_image = cv2.cvtColor((input_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


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