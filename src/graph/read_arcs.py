def read_arcs(file_path, type="original"):
    """
    Reads arcs from a file and returns them as a list of tuples.
    Automatically deduces the type based on the number of fields in each line.

    Args:
        file_path (str): Path to the file containing arc data.
        type (str, optional): Type of arcs to read. Obsolete

    Returns:
        list: A list of tuples representing the arcs.
    """
    arcs = []
    with open(file_path, "r") as file:
        # Read the first line to determine the type
        first_line = file.readline().strip()
        num_fields = len(first_line.split(";"))

        # Seek back to the beginning of the file
        file.seek(0)

        for line in file:
            fields = list(map(int, line.strip().split(";")))
            if num_fields == 4:
                arcs.append(tuple(fields))
            elif num_fields == 5:
                arcs.append(tuple(fields))
            else:
                raise ValueError("Unsupported number of fields in the file.")
    return arcs


def get_arc_name(index):
    """Returns the arc name as a string."""
    return f"Arcs_{index}_1.txt"
