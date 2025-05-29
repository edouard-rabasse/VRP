def read_coordinates(file_path, type="original"):
    coordinates = {}
    last_node = None
    if type not in ["original", "modified"]:
        raise ValueError(
            "[Coordinates] : Invalid type specified. Use 'original' or 'modified'."
        )
    if type == "original":
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                node = int(parts[0])
                x, y = map(float, parts[1:3])
                coordinates[node] = (x, y)
                last_node = node  # The last node is the depot
    elif type == "modified":
        last_node = None
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split(",")
                node = int(parts[0])
                x, y, co_type = map(float, parts[1:4])
                coordinates[node] = (x, y, co_type)
                last_node = node  # The last node is the depot

    return coordinates, last_node


def get_coordinates_name(index):
    """Returns the coordinates name as a string."""
    return f"Coordinates_{index}.txt"
