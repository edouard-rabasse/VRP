"""
VRP coordinate file reading utilities.

Functions for reading node coordinates from VRP instance files
and generating standardized coordinate filenames.
"""


def read_coordinates(
    file_path: str, type: str = "original", keep_service_time: bool = True
) -> tuple[dict, int]:
    """
    Read VRP node coordinates from comma-separated file.

    Args:
        file_path: Path to coordinates file
        type: Coordinate format type ("original" or "modified")
        keep_service_time: Whether to include service time data

    Returns:
        Tuple of (coordinates_dict, depot_node_id)

    Raises:
        ValueError: If invalid type specified
    """

    coordinates = {}
    last_node = None
    if type not in ["original", "modified"]:
        raise ValueError(
            "[Coordinates] : Invalid type specified. Use 'original' or 'modified'."
        )
    with open(file_path, "r") as file:
        try:
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
        except ValueError as e:
            print(f"Error reading file {file_path}: {e}")

    return coordinates, last_node


def get_coordinates_name(index: int) -> str:
    """
    Generate standardized coordinates filename.

    Args:
        index: Instance index number

    Returns:
        Formatted coordinates filename string
    """
    return f"Coordinates_{index}.txt"
