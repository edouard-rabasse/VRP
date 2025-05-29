import matplotlib.pyplot as plt


def plot_routes(arcs, coordinates, depot, output_file, bounds=(-1, 11, -1, 11)):
    # Create a figure and axes with a 10x10 inch size and equal aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    # Remove the borders (spines) from the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove axis ticks and labels for a clean appearance
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Identify nodes that are heads of mode 2 arcs
    mode2_heads = {head for _, head, mode, _ in arcs if mode == 2}
    mode1_heads = {head for _, head, mode, _ in arcs if mode == 1}

    # Plot each arc without adding a legend label (to avoid duplicate legends)
    for tail, head, mode, route_id in arcs:
        x1, y1 = coordinates[tail]
        x2, y2 = coordinates[head]
        linestyle = "-"  # if mode == 1 else '--'
        # Blue for mode 1 and green for mode 2 (if you want to use colors per route, swap accordingly)
        arccolor = (0.0, 1.0, 0.0) if mode == 2 else (0.0, 0.0, 1.0)
        ax.plot(
            [x1, x2],
            [y1, y2],
            linestyle=linestyle,
            color=arccolor,
            linewidth=4,
            zorder=1,
        )

    red = (1.0, 0.0, 0.0)
    for node, (x, y) in coordinates.items():
        if node == depot:
            ax.scatter(x, y, color=red, marker="s", s=60, zorder=2)
        else:
            # Green for nodes that are heads of mode 2 arcs, blue otherwise
            if node in mode2_heads:
                if node in mode1_heads:
                    node_color = (0.0, 0.5, 0.5)
                else:
                    node_color = (0.0, 0.5, 0.0)
            else:
                if node in mode1_heads:
                    node_color = (0.0, 0.0, 0.5)
                else:
                    node_color = (0.5, 0.5, 0.5)

            ax.scatter(x, y, color=node_color, marker="o", s=60, zorder=2)
        # Optionally, you can uncomment the next line to add node labels:
        # ax.text(x + 0.1, y + 0.1, str(node), fontsize=9, color='blue')

    # Turn off the grid
    plt.grid(False)

    # Save the figure to a file and close the plot
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close()
