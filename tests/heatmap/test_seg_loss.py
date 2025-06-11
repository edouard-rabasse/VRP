import torch
import pytest
from src.heatmap import PenalizedSegmentationLoss  # ajuste le chemin selon ton projet


@pytest.fixture
def loss_fn():
    return PenalizedSegmentationLoss(
        false_positive_weight=1.0, false_negative_weight=1.0
    )


def test_perfect_prediction(loss_fn):
    logits = torch.full((1, 1, 4, 4), 10.0)  # ≈ proba ~1 après sigmoid
    mask = torch.ones((1, 1, 4, 4))  # tout est activé
    loss = loss_fn(logits, mask)
    assert loss.item() < 1e-4

    logits = torch.full((1, 1, 4, 4), -10.0)  # ≈ proba ~0 après sigmoid
    mask = torch.zeros((1, 1, 4, 4))
    loss = loss_fn(logits, mask)
    assert loss.item() < 1e-4


def test_false_positives(loss_fn):
    logits = torch.full((1, 1, 4, 4), 10.0)  # prédiction positive partout
    mask = torch.zeros((1, 1, 4, 4))  # aucune activation attendue
    loss = loss_fn(logits, mask)
    assert loss.item() > 0.9  # devrait être proche de 1


def test_false_negatives(loss_fn):
    logits = torch.full((1, 1, 4, 4), -10.0)  # prédiction nulle partout
    mask = torch.ones((1, 1, 4, 4))  # activation attendue partout
    loss = loss_fn(logits, mask)
    assert loss.item() > 0.9  # devrait être proche de 1


def test_mixed_penalties():
    loss_fn = PenalizedSegmentationLoss(
        false_positive_weight=2.0, false_negative_weight=1.0
    )
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])  # deux FP, deux FN
    mask = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    loss = loss_fn(logits, mask)
    # FP: 2 * sigmoid(10) ≈ 2 * 1.0, FN: 1 * (1 - sigmoid(-10)) ≈ 1 * 1.0
    assert abs(loss.item() - 1.5) < 0.01
