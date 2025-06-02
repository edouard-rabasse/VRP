import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap_with_arc(heatmap, bounds, arc_coords, n_samples=20):
    """
    Affiche la heatmap et les points échantillonnés d'un arc pour visualiser l'intersection.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap, extent=bounds, origin="upper", cmap="hot", alpha=0.6)
    ax.set_title("Heatmap + Arc Sampling")

    x_min, x_max, y_min, y_max = bounds
    n_rows, n_cols = heatmap.shape

    def world_to_pixel(x, y):
        fx = (x - x_min) / (x_max - x_min)
        fy = (y_max - y) / (y_max - y_min)
        col = int(fx * n_cols)
        row = int(fy * n_rows)
        return max(0, min(row, n_rows - 1)), max(0, min(col, n_cols - 1))

    def sample_segment(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        for i in range(n_samples):
            t = i / (n_samples - 1)
            yield (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    # Afficher les points échantillonnés
    p1, p2 = arc_coords
    sampled_points = list(sample_segment(p1, p2))

    for x, y in sampled_points:
        row, col = world_to_pixel(x, y)
        ax.plot(x, y, "bo")  # dans le repère réel

    # Trace le segment original
    x_vals, y_vals = zip(*sampled_points)
    ax.plot(x_vals, y_vals, "b--", label="Arc sampled")

    plt.legend()
    plt.grid(True)
    plt.show()


# Paramètres
heatmap = np.zeros((10, 10))
heatmap[6, 3] = 1.0  # pixel chaud

bounds = (0, 10, 0, 10)
arc_coords = ((2.0, 3.0), (3.7, 3.0))  # x approx 2.84 passe-t-il par [6, 3] ?
plot_heatmap_with_arc(heatmap, bounds, arc_coords)
