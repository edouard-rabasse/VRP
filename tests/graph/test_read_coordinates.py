import os
import tempfile
import pytest
from src.graph.read_coordinates import read_coordinates, get_coordinates_name


def test_read_coordinates_original_keep_service():
    content = "1,0.1,0.2,3.5\n2,1.0,1.1,4.2\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    coords, last_node = read_coordinates(
        tmp_path, type="original", keep_service_time=True
    )
    os.remove(tmp_path)

    assert last_node == 2
    assert coords[1] == (0.1, 0.2, 3.5)
    assert coords[2] == (1.0, 1.1, 4.2)
    assert all(len(v) == 3 for v in coords.values())


def test_read_coordinates_original_no_service():
    content = "1,0.1,0.2\n2,1.0,1.1\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    coords, last_node = read_coordinates(
        tmp_path, type="original", keep_service_time=False
    )
    os.remove(tmp_path)

    assert last_node == 2
    assert coords[1] == (0.1, 0.2)
    assert coords[2] == (1.0, 1.1)
    assert all(len(v) == 2 for v in coords.values())


def test_read_coordinates_modified_keep_service():
    content = "1,0.1,0.2,3.5,1.0\n2,1.0,1.1,4.2,2.0\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    coords, last_node = read_coordinates(
        tmp_path, type="modified", keep_service_time=True
    )
    os.remove(tmp_path)

    assert last_node == 2
    assert coords[1] == (0.1, 0.2, 3.5, 1.0)
    assert coords[2] == (1.0, 1.1, 4.2, 2.0)
    assert all(len(v) == 4 for v in coords.values())


def test_read_coordinates_modified_no_service():
    content = "1,0.1,0.2,120,1\n2,1.0,1.1,25,0\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    coords, last_node = read_coordinates(
        tmp_path, type="modified", keep_service_time=False
    )
    os.remove(tmp_path)

    assert last_node == 2
    assert coords[1] == (0.1, 0.2, 1)
    assert coords[2] == (1.0, 1.1, 0)
    assert all(len(v) == 3 for v in coords.values())


def test_read_coordinates_invalid_type():
    with pytest.raises(ValueError):
        read_coordinates("dummy_path", type="invalid")


def test_get_coordinates_name():
    assert get_coordinates_name(123) == "Coordinates_123.txt"
