from src.solver_results.results_dataset import VRPTableDataset
from src.solver_results.vrp_instance import VRPInstance
from src.solver_results.pca_analysis import PCAAnalyzer
from src.solver_results.classification_analysis import ClassificationAnalyzer
from src.solver_results.first_valid_dataset import FirstValidDataset

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


path = "output/resnet_1_1_2e-07_CustomCosts_29_07"
instance = 330

filename = f"instance_{instance}.csv"

df = pd.read_csv(f"{path}/{filename}")


vrp_instance = VRPInstance(instance, df)


def classify(
    vrpinstance: VRPInstance,
    threshold=0.5,
    threshold_class=0.25,
):

    result = pd.DataFrame(
        columns=["instance", "classifier_score_variation", "classifier_score", "valid"]
    )
    for _, row in vrpinstance.get_dataframe().iterrows():
        if (
            -row["classifier_score_variation"] > threshold
            or row["classifier_score"] < threshold_class
        ):
            return row
    return None


def test_classify(threshold_range, instances, threshold_class):
    results = []
    for threshold in threshold_range:
        valid_count = 0
        for instance in instances:
            row = classify(instance, threshold, threshold_class)
            if row is not None:
                if row["valid"]:
                    valid_count += 1
        results.append((threshold, valid_count))

    return results


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


instances = load_vrp_instances(path, range(301, 380))

threshold_range = np.arange(0, 1, 0.1)
for threshold_class in np.arange(0, 0.6, 0.1):
    results = test_classify(threshold_range, instances, threshold_class)
    print("Threshold, Valid Count", threshold_class)
    for threshold, count in results:
        print(f"{threshold:.2f}, {count}")


# def plot_results(results):
#     thresholds, valid_counts = zip(*results)

#     plt.figure(figsize=(10, 6))
#     plt.plot(thresholds, valid_counts, marker="o")
#     plt.title("Valid Count vs Threshold (80 instances)")
#     plt.xlabel("Threshold")
#     plt.ylabel("Valid Count")
#     plt.grid()
#     plt.xticks(threshold_range)
#     plt.yticks(np.arange(0, max(valid_counts) + 1, 5))
#     plt.show()
#     plt.savefig("valid_count_vs_threshold.png", bbox_inches="tight")


# plot_results(results)
