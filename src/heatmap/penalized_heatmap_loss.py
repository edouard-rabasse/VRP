import torch
import torch.nn as nn


class PenalizedSegmentationLoss(nn.Module):
    def __init__(self, false_positive_weight=1.0, false_negative_weight=1.5):
        """
        Custom segmentation loss module penalizing:
        - False positives: predicted heatmap > 0 where mask == 0
        - False negatives: predicted heatmap == 0 where mask == 1

        Args:
            false_positive_weight (float): penalty for predicting activation where none is expected.
            false_negative_weight (float): penalty for missing activation where it's expected.
        """
        super().__init__()
        self.fp_weight = false_positive_weight
        self.fn_weight = false_negative_weight

    def forward(self, seg_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the penalized segmentation loss.

        Args:
            seg_logits (Tensor): model outputs before sigmoid, shape (N, 1, H, W)
            masks (Tensor): ground truth binary masks, shape (N, 1, H, W)

        Returns:
            Tensor: scalar loss value
        """
        probs = torch.sigmoid(seg_logits)

        false_positives = (1 - masks) * probs  # model says "yes", mask says "no"
        false_negatives = masks * (1 - probs)  # model says "no", mask says "yes"

        loss = (
            self.fp_weight * false_positives.mean()
            + self.fn_weight * false_negatives.mean()
        )
        return loss
