import pytest
from src.utils.config_utils import load_selection_config
from unittest.mock import MagicMock


def test_load_selection_config_range():
    mock_cfg = MagicMock()
    mock_cfg.selection.type = "range"
    mock_cfg.selection.value = (2, 5)
    result = load_selection_config(mock_cfg)
    assert list(result) == [2, 3, 4]  # range(2,5) donne 2,3,4


def test_load_selection_config_indices():
    mock_cfg = MagicMock()
    mock_cfg.selection.type = "indices"
    mock_cfg.selection.value = [10, 20, 30]
    result = load_selection_config(mock_cfg)
    assert result == [10, 20, 30]


def test_load_selection_config_unknown_type():
    mock_cfg = MagicMock()
    mock_cfg.selection.type = "unknown"
    mock_cfg.selection.value = None
    with pytest.raises(ValueError, match="Unknown selection type"):
        load_selection_config(mock_cfg)
