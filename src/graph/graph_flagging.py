import torch
from PIL import Image
from omegaconf import DictConfig

from src.graph import generate_plot_from_dict, read_coordinates, read_arcs
from src.models import load_model
from src.visualization import get_heatmap
from src.transform import image_transform_test
from src.graph import HeatmapAnalyzer


def flag_graph_from_instance(
    instance_number: int,
    model: torch.nn.Module,
    cfg: DictConfig,
    device: str = "cuda",
    return_weighted_sum: bool = False,
    top_n_arcs: int | None = None,
    threshold: float = 0.5,
) -> tuple:
    """
    Load arc and coordinate data for a given instance number, and return flagged elements.

    Args:
        instance_number (int): The instance ID.
        model (torch.nn.Module): Trained model.
        cfg (DictConfig): Configuration for heatmap generation.
        device (str): Torch device.
        return_weighted_sum (bool): Whether to return the weighted sum.

    Returns:
        tuple: (flagged_arcs, flagged_coordinates)
    """
    coord_path = f"MSH/MSH/instances/Coordinates_{instance_number}.txt"
    arc_path = f"MSH/MSH/results/configuration1/Arcs_{instance_number}_1.txt"

    arcs = read_arcs(arc_path)
    coordinates, depot = read_coordinates(coord_path)

    return flag_graph_from_data(
        arcs,
        coordinates,
        depot,
        model,
        cfg,
        device,
        return_weighted_sum=return_weighted_sum,
        top_n_arcs=top_n_arcs,
        threshold=threshold,
    )


def flag_graph_from_data(
    arcs: list,
    coordinates: dict,
    depot: int,
    model: torch.nn.Module,
    cfg: DictConfig,
    device: str = "cuda",
    return_weighted_sum: bool = False,
    top_n_arcs: int | None = None,
    threshold: float = 0.5,
) -> tuple:
    """
    Generate a plot from graph data, forward it through the model, and flag arcs/coordinates.

    Args:
        arcs (list): List of arcs.
        coordinates (dict): Node coordinates.
        model (torch.nn.Module): Trained model.
        cfg (DictConfig): Configuration for heatmap generation.
        device (str): Torch device.
        return_weighted_sum (bool): Whether to return the weighted sum.

    Returns:
        tuple: (flagged_arcs, flagged_coordinates)
    """
    image = generate_plot_from_dict(
        arcs, coordinates, depot=depot, bounds=tuple(cfg.plot.bounds)
    )
    input_tensor = (
        image_transform_test()(Image.fromarray(image)).unsqueeze(0).to(device)
    )

    return flag_graph_from_tensor(
        input_tensor,
        arcs,
        coordinates,
        model,
        cfg,
        device,
        return_weighted_sum=return_weighted_sum,
        top_n_arcs=top_n_arcs,
        threshold=threshold,
    )


def flag_graph_from_tensor(
    input_tensor: torch.Tensor,
    arcs: list,
    coordinates: dict,
    model: torch.nn.Module,
    cfg: DictConfig,
    device: str = "cuda",
    return_weighted_sum: bool = False,
    top_n_arcs: int | None = None,
    threshold: float = 0.5,
) -> tuple:
    """
    Compute heatmap from a tensor input and flag important arcs/nodes.

    Args:
        input_tensor (torch.Tensor): The input image tensor.
        arcs (list): List of arcs.
        coordinates (dict): Node coordinates.
        model (torch.nn.Module): Trained model.
        cfg (DictConfig): Heatmap configuration.
        device (str): Torch device.
        return_weighted_sum (bool): Whether to return the weighted sum.

    Returns:
        tuple: (flagged_arcs, flagged_coordinates)
    """
    heatmap = get_heatmap(
        model=model,
        method=cfg.heatmap.method,
        input_tensor=input_tensor,
        args=cfg.heatmap.args,
        device=device,
    )

    analyzer = HeatmapAnalyzer(
        heatmap=heatmap, arcs=arcs, coordinates=coordinates, threshold=threshold
    )
    return analyzer.reverse_heatmap(
        return_weighted_sum=return_weighted_sum, top_n_arcs=top_n_arcs
    )
