import matplotlib.pyplot as plt


def plot_routes(
    arcs,
    coordinates,
    depot,
    output_file,
    bounds=(-1, 11, -1, 11),
    route_type="original",  # "original" or "modified"
    show_labels=False,
):
    """
    Plot routes given arcs and coordinates.

    Parameters:
    - arcs: List of tuples (tail, head, mode, route_id) or (tail, head, mode, route_id, arc_type)
    - coordinates: Dict of node -> (x, y) or node -> (x, y, co_type)
    - depot: ID of the depot node
    - output_file: Path to save the plot
    - bounds: Tuple (xmin, xmax, ymin, ymax)
    - route_type: "original" or "modified"
    - show_labels: whether to display node labels
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.margins(0)

    # Clean plot look
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.grid(False)

    # Determine if arc_type is present
    arc_has_type = len(arcs[0]) == 5

    # Collect heads per mode for node coloring (only used for "original" style)
    mode2_heads = {arc[1] for arc in arcs if arc[2] == 2}
    mode1_heads = {arc[1] for arc in arcs if arc[2] == 1}

    # Plot arcs
    for arc in arcs:
        tail, head, mode, route_id = arc[:4]
        arc_type = arc[4] if arc_has_type else 0

        x1, y1 = coordinates[tail][:2]
        x2, y2 = coordinates[head][:2]

        linestyle = "-" if arc_type == 0 else ":"
        arccolor = (0.0, 1.0, 0.0) if mode == 2 else (0.0, 0.0, 1.0)

        ax.plot(
            [x1, x2],
            [y1, y2],
            linestyle=linestyle,
            color=arccolor,
            linewidth=4,
            zorder=1,
        )

    # Plot nodes
    for node, coord in coordinates.items():
        x, y = coord[:2]
        co_type = coord[2] if len(coord) > 2 else None

        if node == depot:
            color = (1.0, 0.0, 0.0)
            marker = "s"
        else:

            if node in mode2_heads:
                color = (0.0, 0.5, 0.5) if node in mode1_heads else (0.0, 0.5, 0.0)
            else:
                color = (0.0, 0.0, 0.5) if node in mode1_heads else (0.5, 0.5, 0.5)
            if route_type == "original":
                marker = "o"
            else:
                marker = "s" if co_type == 1 else "o"

        ax.scatter(x, y, color=color, marker=marker, s=60, zorder=2)

        if show_labels:
            ax.text(x + 0.1, y + 0.1, str(node), fontsize=9, color="blue")

    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close()
