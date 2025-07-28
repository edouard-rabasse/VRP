from omegaconf import DictConfig
import numpy as np

import torch

from PIL import Image


class Scoring:
    def __init__(self, cfg: DictConfig, model: torch.nn.Module, device: torch.device):
        self.cfg = cfg
        self.model = model
        self.device = device

    def compute_all_scores(
        self,
        coords: dict,
        arcs: list,
        depot: int,
        heatmap: np.ndarray,
        input_tensor: torch.Tensor,
    ) -> tuple:
        try:
            scores = {}

            scores["classifier_score"] = self._score(input_tensor)
            scores["entropy_score"] = self._entropy_heatmap(heatmap)
            scores["top_arc_value"] = self._top_arcs(arcs, index=4, number=1)
            scores["top_3_arcs"] = self._top_arcs(arcs, index=4, number=3)
            return scores
        except Exception as e:
            raise ValueError(f"Failed to compute all scores: {e}") from e

    def _score(self, input_tensor: torch.Tensor) -> float:

        try:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(input_tensor)

                # score = torch.sigmoid(out).squeeze().cpu()[1].item()
                score = torch.softmax(logits, dim=1)[0, 1].item()
                print(f"[Debug] Exact score: {score:.20f}")
            return score
        except Exception as e:
            print(f"Error occurred while scoring the input tensor: {e}")

    def _entropy_heatmap(
        self,
        heatmap: np.ndarray,
    ) -> float:
        """Compute the entropy of the heatmap as a score."""
        try:
            if heatmap is None:
                return 0.0
            # Normalize the heatmap to avoid division by zero
            heatmap = heatmap / np.sum(heatmap) if np.sum(heatmap) > 0 else heatmap
            # Compute entropy as a score
            score = -np.sum(heatmap * np.log(heatmap + 1e-10))
            return score
        except Exception as e:
            raise ValueError(f"Failed to compute entropy from heatmap: {e}") from e

    def _top_arcs(self, flagged_arcs: list, index: int = 4, number: int = 3) -> list:
        """
        Extract the top N arcs from the flagged arcs based on their weights.
        Args:
            flagged_arcs (list): List of flagged arcs.
            index (int): Index of the weight in the arc tuple.
        Returns:
            list: List of top N arcs.
        """
        try:
            sorted_arcs = sorted(flagged_arcs, key=lambda x: x[index], reverse=True)
            average = 0
            for i in range(number):
                if i < len(sorted_arcs):
                    average += sorted_arcs[i][index]
            average /= number
            return average
        except Exception as e:
            raise ValueError(f"Failed to extract top arc value: {e}") from e
