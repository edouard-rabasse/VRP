def read_arcs(file_path, type="original", number_of_fields=None):
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
            if number_of_fields is None:
                number_of_fields = num_fields
            if num_fields == 4:
                arcs.append(tuple(fields[:number_of_fields]))
            elif num_fields == 5:
                arcs.append(tuple(fields[:number_of_fields]))
            else:
                raise ValueError("Unsupported number of fields in the file.")

    return arcs


def binarize_arcs(arcs, threshold=0, index=3):
    """
    Binarizes the arcs based on a threshold value.

    Args:
        arcs (list): List of arcs, where each arc is a tuple.
        threshold (float): Threshold value for binarization.
        index (int): Index of the field to apply the threshold.

    Returns:
        list: List of arcs with binarized values.
    """
    binarized_arcs = []
    for arc in arcs:
        if len(arc) > index:
            binarized_value = 1 if arc[index] > threshold else 0
            binarized_arcs.append(arc[:index] + (binarized_value,))
    return binarized_arcs


def get_arc_name(index, suffix: int | str = 1) -> str:
    """Returns the arc name as a string."""
    return f"Arcs_{index}_{suffix}.txt"


if __name__ == "__main__":
    # Example usage
    arcs = read_arcs("MSH/MSH/results/configuration1/Arcs_1_1.txt")
    print(arcs)

    binarized_arcs = binarize_arcs(arcs, threshold=-1, index=4)
    print(binarized_arcs)

    arc_name = get_arc_name(1, suffix=2)
    print(arc_name)
