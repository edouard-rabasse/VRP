import cv2
import numpy as np
import torch


def resize_heatmap(heatmap, target_size):
    """Resize the heatmap to match the target size."""
    heatmap_resized = cv2.resize(heatmap, target_size)
    heatmap_resized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))
    
    return heatmap_resized

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
    
    elif method == "multi_task":
        with torch.no_grad():
            cls_logits, seg_logits = model(input_tensor)
            heatmap = seg_logits[0, 0].cpu().numpy()  # Assuming the first channel is the one of interest
            heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[3]))
        
    else:
        raise ValueError("Unknown method: {}".format(method))

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
    _, heatmap_binary = cv2.threshold(heatmap_resized, 0.5 * 255, 255, cv2.THRESH_BINARY)
    
    # Compute intersection
    intersection = cv2.bitwise_and(heatmap_binary, mask)
    
    return intersection