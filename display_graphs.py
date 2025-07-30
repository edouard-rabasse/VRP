from src.solver_results.results_dataset import VRPTableDataset
from src.solver_results.vrp_instance import VRPInstance
from src.solver_results.pca_analysis import PCAAnalyzer
from src.solver_results.classification_analysis import ClassificationAnalyzer
from src.solver_results.first_valid_dataset import FirstValidDataset

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


path = "output/resnet_1_1_2e-07_CustomCosts_29_07"


def load_vrp_instances(path, instance_range):
    """Load VRP instances from CSV files."""
    instances = []

    for i in instance_range:
        filename = f"instance_{i}.csv"
        try:
            df = pd.read_csv(f"{path}/{filename}")
            # print(f"Processing {filename} with {len(df)} rows.")
            instance = VRPInstance(i, df)
            instances.append(instance)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return instances


def compute_mean_df_by_iteration(instances):
    all_dfs = []
    has_been_valid_flags = {}

    for instance in instances:
        df = instance.get_dataframe()
        df["instance"] = instance.instance_number  # Optional: keep track of source
        df["has_been_valid"] = df["valid"].expanding().max().astype(bool)
        all_dfs.append(df)

    # Concatenate all dataframes
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Group by iteration and compute mean (exclude non-numeric columns automatically)
    mean_df = full_df.groupby("iter").mean(numeric_only=True).reset_index()

    valid_counts = (
        full_df.groupby("iter")["valid"].sum().reset_index(name="number_of_valid")
    )

    number_of_has_been_valid = (
        full_df.groupby("iter")["has_been_valid"]
        .sum()
        .reset_index(name="number_of_has_been_valid")
    )

    first_valid = (
        full_df.groupby("iter")["first_time_valid"]
        .sum()
        .reset_index(name="number_of_first_valid")
    )

    # Merge with mean_df
    mean_df = mean_df.merge(valid_counts, on="iter")
    mean_df = mean_df.merge(number_of_has_been_valid, on="iter")
    mean_df = mean_df.merge(first_valid, on="iter")

    return mean_df


instances = load_vrp_instances(path, range(301, 380))
mean_df = compute_mean_df_by_iteration(instances)

instance = 330
columns = [
    "number_of_violations",
    "classifier_score",
    # "entropy_score",
    # "top_arc_value",
    # "top_3_arcs",
    # "entropy_previous",
    # "classifier_score_previous",
    # "entropy_variation",
    "classifier_score_variation",
    # "top_arc_variation",
    # "top_3_arcs_variation",
    "number_of_valid",
    "number_of_has_been_valid",
    "number_of_first_valid",
]

filename = f"instance_{instance}.csv"

df = pd.read_csv(f"{path}/{filename}")


vrp_instance = VRPInstance(instance, df)

df = vrp_instance.get_dataframe()


def plot_multiple_y(df, columns, x_col="iter", title="Multiple Y-Axis Plot"):
    """
    Plot multiple columns from a DataFrame on the same X-axis, using:
    - shared Y-axis for columns containing "score"
    - separate Y-axes for others
    - labels in a legend box

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list of str): List of column names to plot.
        x_col (str): Column for the X-axis.
        title (str): Plot title.
    """
    if not columns:
        raise ValueError("You must provide at least one column to plot.")

    fig, ax_main = plt.subplots()
    fig.suptitle(title)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    shared_ax = None
    shared_ax_2 = None
    other_axes = []
    lines = []
    labels = []

    for i, col in enumerate(columns):
        color = color_cycle[i % len(color_cycle)]

        if "score" in col.lower() and "entropy" not in col.lower():
            if shared_ax is None:
                shared_ax = ax_main
                shared_ax.set_ylabel(col, color=color)
                # shared_ax.spines["left"].set_position(("outward", 0.5))
            else:
                shared_ax = shared_ax
            (line,) = shared_ax.plot(df[x_col], df[col], label=col, color=color)
            shared_ax.tick_params(axis="y", labelcolor=color)
        # elif "valid" in col.lower():
        #     if shared_ax_2 is None:
        #         shared_ax_2 = ax_main
        #         shared_ax_2.set_ylabel(col, color=color)
        #         shared_ax_2.spines["right"].set_position(("outward", 0.5))
        #     else:
        #         shared_ax_2 = shared_ax_2
        #     (line,) = shared_ax_2.plot(df[x_col], df[col], label=col, color=color)
        #     shared_ax_2.tick_params(axis="y", labelcolor=color)

        else:
            ax_new = ax_main.twinx()
            ax_new.spines["right"].set_position(("outward", 60 * len(other_axes)))
            ax_new.set_ylabel(col, color=color)
            (line,) = ax_new.plot(
                df[x_col], df[col], label=col, color=color, linestyle="--"
            )
            ax_new.tick_params(axis="y", labelcolor=color)
            other_axes.append(ax_new)

        lines.append(line)
        labels.append(col)

    ax_main.set_xlabel(x_col)

    # Add legend in a box
    ax_main.legend(
        lines, labels, loc="upper right", frameon=True, bbox_to_anchor=(1.05, 1.0)
    )

    fig.tight_layout()
    plt.show()

    # save the plot
    # fig.savefig(f"plot_mean.png", bbox_inches="tight")


plot_multiple_y(mean_df, columns, x_col="iter", title="mean")
