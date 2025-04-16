import torch.nn as nn
import torch

def evaluate_model_cls(model, test_loader, device="cpu"):
    """
    prints metrics for the classification model
    F1 score, accuracy, precision, recall
    confusion matrix"""
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, masks in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays for sklearn functions
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Print classification report
    print(classification_report(all_labels, all_preds))
    
    # Print confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

def evaluate_seg_cls(model, test_loader,method, device="cpu"):
    """
    prints metrics for the segmentation model, using heatmap
    TODO : WIP
    """
    from .visualization import get_heatmap
    from sklearn.metrics import classification_report, confusion_matrix
    for images, labels, masks in test_loader:
        heatmap = get_heatmap(method, model, input_tensor=images, args=None, device=device)
        print(heatmap.shape)
    

