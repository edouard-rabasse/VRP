def load_set_arc(arcs: list[tuple], number: int = 4):
    set_arcs = set()

    for arc in arcs:
        # add the number of fields
        set_arcs.add(arc[:number])
    return set_arcs


def test_load_set_arc():
    arcs = [(1, 2, 3, 4), (5, 6, 7, 8), (1, 2, 3, 4), (9, 10, 11, 12)]  # Duplicate
    result = load_set_arc(arcs)
    assert len(result) == 3, "Should return unique arcs"
    assert (1, 2, 3, 4) in result
    assert (5, 6, 7, 8) in result
    assert (9, 10, 11, 12) in result
    print("Test passed!")


if __name__ == "__main__":
    test_load_set_arc()
    # Example usage
    arcs = [(1, 2, 3, 4), (5, 6, 7, 8), (1, 2, 3, 4), (9, 10, 11, 12)]
    unique_arcs = load_set_arc(arcs)
    print(unique_arcs)  # Output: {(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)}
