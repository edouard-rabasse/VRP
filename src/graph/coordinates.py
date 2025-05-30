def read_coordinates(file_path, type="original", keep_service_time=False):
    coordinates = {}
    last_node = None
    if type not in ["original", "modified"]:
        raise ValueError(
            "[Coordinates] : Invalid type specified. Use 'original' or 'modified'."
        )
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            node = int(parts[0])
            x, y = map(float, parts[1:3])
            if type == "original":
                if keep_service_time:
                    service_time = float(parts[3]) if len(parts) > 3 else 0.0
                    coordinates[node] = (x, y, service_time)
                else:
                    coordinates[node] = (x, y)
            elif type == "modified":
                if keep_service_time:
                    service_time = float(parts[3]) if len(parts) > 3 else 0.0
                    co_type = float(parts[4]) if len(parts) > 4 else 0.0
                    coordinates[node] = (x, y, service_time, co_type)
                else:
                    co_type = float(parts[4]) if len(parts) > 4 else 0.0
                    coordinates[node] = (x, y, co_type)

            last_node = node  # The last node is the depot

    return coordinates, last_node


def get_coordinates_name(index):
    """Returns the coordinates name as a string."""
    return f"Coordinates_{index}.txt"
