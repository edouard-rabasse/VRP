import numpy as np
import torch
from .get_attr import recursive_getattr
import os
from PIL import Image
from src.transform import image_transform_test
from collections import defaultdict

from ..graph.HeatmapAnalyzer import HeatmapAnalyzer


from ..graph import get_arc_name, get_coordinates_name, read_arcs, read_coordinates


import cv2
import torchvision.transforms.functional as TF

from .get_heatmap import get_heatmap
from .show_mask_on_image import show_mask_on_image
from ..graph.generate_plot import generate_plot_from_dict
from src.transform import ProportionalThresholdResize

from omegaconf import DictConfig


class HeatmapMetric:
    """
    Class to handle heatmap metrics, including loading the model, processing images,
    and computing losses.
    """

    def __init__(self, cfg: DictConfig, model: torch.nn.Module):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        print(f"[Viz] Model loaded: {cfg.model.name}")

    def _init_file(self, fname: str) -> tuple:
        """
        Helper function to initialize the file by loading the image and mask.
        Returns the tensor of the image and the mask.
        """

        self.coordinates, self.arcs_in, self.number = self._get_in_arcs_and_coords(
            fname
        )
        self.arcs_out = self._get_out_arcs(self.number)

        self.arcs_diff, self.common_arcs = self._get_arc_diff(
            self.arcs_in, self.arcs_out
        )
        self.tensor_img = self._get_image_tensor(fname)
        self.tensor_img = self.tensor_img.to(self.device)

        self.heatmap = self._get_heatmap(self.tensor_img)

    def _get_image_tensor(self, fname: str) -> tuple:
        """
        Helper function to get the the tensor of the image and the mask from the filename.
        """

        img = generate_plot_from_dict(self.arcs_in, self.coordinates, self.number)
        input_tensor = (
            image_transform_test()(Image.fromarray(img)).unsqueeze(0).to(self.device)
        )
        return input_tensor

    def _get_heatmap(self, t_img: torch.Tensor) -> np.ndarray:
        """
        Helper function to compute the heatmap from the tensor of the image.
        """
        heatmap = get_heatmap(
            self.cfg.heatmap.method,
            self.model,
            t_img,
            self.cfg.heatmap.args,
            device=self.device,
        )
        return heatmap

    def _get_in_arcs_and_coords(self, fname: str, number_of_fields=3) -> tuple:
        """
        Helper function to get the arcs and coordinates from the filename.
        """
        number = int(fname.split(".")[0].split("_")[1])
        coordinates_p = os.path.join(
            self.cfg.arcs.coord_in_dir, get_coordinates_name(number)
        )
        arcs_p = os.path.join(self.cfg.arcs.arcs_in_dir, get_arc_name(number))
        coordinates, _ = read_coordinates(coordinates_p, keep_service_time=True)
        arcs = read_arcs(arcs_p, number_of_fields=number_of_fields)
        return coordinates, arcs, number

    def _get_out_arcs(
        self, number: int, suffix: int | str = 1, number_of_fields=3
    ) -> list:
        """
        Helper function to get the output arcs and coordinates paths.
        """
        arcs_out = os.path.join(
            self.cfg.arcs.arcs_refined_dir, get_arc_name(number, suffix=suffix)
        )
        arcs_out = read_arcs(arcs_out, number_of_fields=number_of_fields)
        return arcs_out

    def _get_arc_diff(self, arcs_in, arcs_out) -> list:
        """
        Helper function to compute the difference between input and output arcs.
        Returns a list of arcs that are in the input but not in the output, and a list of common arcs.
        """
        arcs_in_set = set(arcs_in)
        arcs_out_set = set(arcs_out)
        diff = list(arcs_in_set - arcs_out_set)
        common_arcs = list(arcs_in_set & arcs_out_set)
        return diff, common_arcs

    def _process_one(self, threshold: float = 0.5):
        """
        From the heatmapanalyzer, find the arcs
        """
        analyzer = HeatmapAnalyzer(
            heatmap=self.heatmap,
            coordinates=self.coordinates,
            arcs=self.arcs_in,
            threshold=threshold,
            n_samples=self.cfg.arcs.n_samples,
        )
        arcs_with_zone, updated_coords = analyzer.reverse_heatmap()

        ## only keep the first 3 fields

        number_arcs = len(arcs_with_zone)

        arcs_with_zone = [arc[:3] for arc in arcs_with_zone if arc[3] > 0]

        ## Compare the arcs with the common arcs
        common_arcs = [arc[:3] for arc in self.common_arcs]
        arcs_diff = [arc[:3] for arc in self.arcs_diff]

        number_common = len(common_arcs)
        number_diff = len(arcs_diff)

        false_positive = list(set(arcs_with_zone) & set(common_arcs))

        true_positive = list(set(arcs_with_zone) & set(arcs_diff))

        FP = len(false_positive)
        TP = len(true_positive)
        FN = number_diff - TP
        TN = number_common - FP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0

        return {
            "nb_diff": number_diff,
            "nb_common": number_common,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": false_positive_rate,
        }

    def process_image(self, fname: str, list_threshold) -> list:
        """
        Processes a single image, computes the heatmap, and calculates the loss.
        Args:
            fname (str): The filename or path of the image to process.
            list_threshold (list): List of thresholds to apply for heatmap analysis.
        Returns:
            list: A list of dictionaries containing the metrics for each threshold. the metrics include:
            - nb_diff: Number of arcs in the difference
            - nb_common: Number of common arcs
            - TP: True Positives
            - FP: False Positives
            - FN: False Negatives
            - TN: True Negatives
            - precision: Precision of the heatmap
            - recall: Recall of the heatmap
            - f1: F1 score of the heatmap
            - false_positive_rate: False Positive Rate of the heatmap
        """
        self._init_file(fname)

        results = []
        for threshold in list_threshold:
            metrics = self._process_one(threshold)
            metrics["filename"] = fname
            metrics["threshold"] = threshold
            results.append(metrics)

        return results
