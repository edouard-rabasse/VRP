def recursive_getattr(model, attr_path):
    """Get nested attribute from a module using dot notation."""
    attrs = attr_path.split(".")
    for attr in attrs:
        if attr.isdigit():
            model = model[int(attr)]
        else:
            model = getattr(model, attr)
    return model
