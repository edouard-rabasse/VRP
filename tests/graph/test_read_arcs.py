import os
import tempfile
from src.graph.read_arcs import read_arcs, get_arc_name


def test_read_arcs_original():
    content = "1;2;1;0\n3;4;2;1\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    arcs = read_arcs(tmp_path)
    os.remove(tmp_path)

    assert arcs == [(1, 2, 1, 0), (3, 4, 2, 1)]
    assert all(len(arc) == 4 for arc in arcs)


def test_read_arcs_modified():
    content = "1;2;1;0;0\n3;4;2;1;1\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    arcs = read_arcs(tmp_path)
    os.remove(tmp_path)

    assert arcs == [(1, 2, 1, 0, 0), (3, 4, 2, 1, 1)]
    assert all(len(arc) == 5 for arc in arcs)


def test_get_arc_name():
    assert get_arc_name(42) == "Arcs_42_1.txt"
