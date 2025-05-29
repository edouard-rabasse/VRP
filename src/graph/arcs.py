def read_arcs(file_path, type="original"):
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
