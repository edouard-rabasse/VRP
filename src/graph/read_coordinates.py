# This module provides functions to read node coordinates from a file and to generate coordinate file names.
# Functions:
#     read_coordinates(file_path, type="original", keep_service_time=True) -> tuple[dict, int]:
#         Reads coordinates from a file and returns them as a dictionary along with the last node ID (depot).
#         Supports two types of coordinate formats: "original" and "modified".
#         Optionally includes service time and coordinate type information.
#     get_coordinates_name(index):
#         Returns the coordinates file name as a string based on the given index.
def read_coordinates(
    file_path, type="original", keep_service_time=True
) -> tuple[dict, int]:
    """Reads coordinates from a file and returns them as a dictionary.

    Args:
        file_path (str): Path to the coordinates file.
        type (str, optional): Type of coordinates to read. Defaults to "original".
        keep_service_time (bool, optional): Whether to keep service time information. Defaults to False.

    Raises:
        ValueError: If an invalid type is specified.

    Returns:
        tuple: A tuple containing:
            - coordinates (dict): dict: {node_id: (x, y, [service_time], [co_type])}
            - last_node (int): The last node ID (depot)
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


def get_coordinates_name(index):
    """Returns the coordinates name as a string."""
    return f"Coordinates_{index}.txt"
