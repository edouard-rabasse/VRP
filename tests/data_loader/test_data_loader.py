import pytest
from unittest.mock import MagicMock, patch, ANY
from src.data_loader import load_data


@pytest.fixture
def mock_cfg_mask():
    cfg = MagicMock()
    cfg.data.loader = "mask"
    cfg.data.train_original_path = "train_orig"
    cfg.data.test_original_path = "test_orig"
    cfg.data.train_modified_path = "train_mod"
    cfg.data.test_modified_path = "test_mod"
    cfg.data.train_mask_path = "train_mask"
    cfg.data.test_mask_path = "test_mask"
    cfg.batch_size = 4
    cfg.image_size = [224, 224]
    cfg.mask_shape = [224, 224]
    cfg.num_workers = 0
    cfg.data.augment = False
    cfg.data.selection.type = "range"
    cfg.data.selection.value = (0, 2)
    return cfg


@pytest.fixture
def mock_cfg_graph():
    cfg = MagicMock()
    cfg.data.loader = "graph"
    cfg.data.orig_arcs_folder = "orig_arcs"
    cfg.data.mod_arcs_folder = "mod_arcs"
    cfg.data.coords_folder = "coords"
    cfg.data.bounds = [-1, 11, -1, 11]
    cfg.data.pixel_size = 1
    cfg.data.mask_method = "default"
    cfg.batch_size = 2
    cfg.image_size = [224, 224]
    cfg.mask_shape = [224, 224]
    cfg.plot.train_ratio = 0.7
    cfg.plot.random_state = 123
    cfg.data.selection.type = "range"
    cfg.data.selection.value = [1, 3]
    return cfg


# @patch("src.data_loader.load_selection_config", return_value=range(0, 2))
@patch("src.data_loader.load_data_train_test")
@patch("src.data_loader_mask._get_filenames_by_index")
def test_load_data_mask_calls_load_data_train_test(
    mock_get_filenames,
    mock_load_data_train_test,
    # mock_load_selection_config,
    mock_cfg_mask,
):
    # Simuler la récupération des fichiers pour éviter FileNotFoundError
    mock_get_filenames.side_effect = lambda dir_path, indices: [
        f"{dir_path}/img_{i}.png" for i in indices
    ]
    mock_load_data_train_test.return_value = ("train_loader_mock", "test_loader_mock")

    train_loader, test_loader = load_data(mock_cfg_mask)

    mock_load_data_train_test.assert_called_once()
    assert train_loader == "train_loader_mock"
    assert test_loader == "test_loader_mock"


@patch("src.data_loader.load_selection_config", return_value=[1, 2, 3])
@patch("src.data_loader.DataLoader")
def test_load_data_graph_calls_vrpgraphdataset_and_split(
    mock_dataloader,
    mock_load_selection_config,
    mock_cfg_graph,
):
    with patch("src.data_loader.VRPGraphDataset") as mock_dataset_cls, patch(
        "src.data_loader.split_graph_dataset"
    ) as mock_split:

        mock_dataset = MagicMock()
        mock_dataset.instances = [("1", "orig_path", "mod_path", "coord_path")]
        mock_dataset.__len__.return_value = 1
        mock_dataset_cls.return_value = mock_dataset

        mock_train_ds = MagicMock()
        mock_train_ds.__len__.return_value = 1
        mock_test_ds = MagicMock()
        mock_test_ds.__len__.return_value = 1
        mock_split.return_value = (mock_train_ds, mock_test_ds)

        # DataLoader mock retourne une chaîne pour faciliter la vérification
        mock_dataloader.side_effect = lambda ds, **kwargs: f"Dataloader({ds})"

        train_loader, test_loader = load_data(mock_cfg_graph)

        mock_dataset_cls.assert_called_once_with(
            orig_arcs_folder=mock_cfg_graph.data.orig_arcs_folder,
            mod_arcs_folder=mock_cfg_graph.data.mod_arcs_folder,
            coords_folder=mock_cfg_graph.data.coords_folder,
            bounds=tuple(mock_cfg_graph.data.bounds),
            pixel_size=mock_cfg_graph.data.pixel_size,
            mask_method=mock_cfg_graph.data.mask_method,
            image_transform=ANY,
            mask_transform=ANY,
            valid_range=mock_load_selection_config.return_value,
        )

        mock_split.assert_called_once_with(
            mock_dataset,
            train_ratio=mock_cfg_graph.plot.train_ratio,
            seed=mock_cfg_graph.plot.random_state,
        )

        assert train_loader == f"Dataloader({mock_train_ds})"
        assert test_loader == f"Dataloader({mock_test_ds})"


def test_load_data_invalid_loader_type_raises(mock_cfg_mask):
    mock_cfg_mask.data.loader = "unknown_loader"
    with pytest.raises(ValueError, match="Unknown data.loader"):
        load_data(mock_cfg_mask)
