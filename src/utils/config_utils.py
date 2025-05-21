def load_selection_config(config):
    """
    Load the selection configuration from the provided config dictionary.
    The selection configuration should specify a type and values."""

    selection_type = config.selection.type
    values = config.selection.value

    if selection_type == "range":
        start, end = values
        selected_values = range(start, end)
    elif selection_type == "indices":
        selected_values = values
    else:
        raise ValueError("Unknown selection type")

    return selected_values
