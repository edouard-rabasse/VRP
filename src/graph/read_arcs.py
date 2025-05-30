def read_arcs(file_path, type="original"):
    """
    Reads arcs from a file and returns them as a list of tuples.

    Args:
        file_path (str): Path to the file containing arc data.
        type (str, optional): Type of arcs to read. Defaults to "original".

    Returns:
        list: A list of tuples representing the arcs.
    """
    arcs = []
    with open(file_path, "r") as file:
        if type == "original":
            for line in file:
                tail, head, mode, route_id = map(int, line.strip().split(";"))
                arcs.append((tail, head, mode, route_id))
        elif type == "modified":
            for line in file:
                tail, head, mode, route_id, arc_type = map(int, line.strip().split(";"))
                arcs.append((tail, head, mode, route_id, arc_type))
    return arcs


def get_arc_name(index):
    """Returns the arc name as a string."""
    return f"Arcs_{index}_1.txt"
