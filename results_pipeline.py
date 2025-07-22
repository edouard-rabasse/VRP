from src.solver_results.results_dataset import VRPTableDataset
from src.solver_results.vrp_instance import VRPInstance
from src.solver_results.pca_analysis import PCAAnalyzer
from src.solver_results.classification_analysis import ClassificationAnalyzer

import pandas as pd
import numpy as np


def main():
    """Main pipeline function."""
    path = "output/resnet_1_1_2e-07_version_cedar"

    # Load data
    VRPInstances = load_vrp_instances(path, range(1001, 1086))

    # Define features
    feature_cols = [
        "entropy_score",
        "classifier_score",
        "iter",
        "entropy_variation",
        "classifier_score_variation",
        "top_arc_value",
        "top_3_arcs",
        "top_arc_variation",
        "top_3_arcs_variation",
    ]

    # Create dataset
    dataset = VRPTableDataset(
        vrp_instances=VRPInstances,
        feature_cols=feature_cols,
        target_col="first_time_valid",
    )

    X, Y = dataset.get_sklearn_inputs()

    # Run analyses
    run_pca_analysis(X, Y, feature_cols)
    run_classification_analysis(X, Y, feature_cols)


def load_vrp_instances(path, instance_range):
    """Load VRP instances from CSV files."""
    instances = []

    for i in instance_range:
        filename = f"instance_{i}.csv"
        try:
            df = pd.read_csv(f"{path}/{filename}")
            print(f"Processing {filename} with {len(df)} rows.")
            instance = VRPInstance(i, df)
            instances.append(instance)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return instances


def run_pca_analysis(X, Y, feature_cols):
    """Run PCA analysis."""
    print("\n" + "=" * 50)
    print("PCA ANALYSIS")
    print("=" * 50)

    pca_analyzer = PCAAnalyzer()
    pca_analyzer.plot_analysis(X, Y, feature_cols)

    # Find optimal components
    optimal_components, results_df = pca_analyzer.find_optimal_components(X, Y)

    return pca_analyzer


def run_classification_analysis(X, Y, feature_cols):
    """Run classification analysis."""
    print("\n" + "=" * 50)
    print("CLASSIFICATION ANALYSIS")
    print("=" * 50)

    classifier_analyzer = ClassificationAnalyzer()

    # Analyze with original features
    results_original = classifier_analyzer.run_analysis(
        X, Y, feature_cols, "Original Features"
    )

    # Analyze with PCA features
    pca_analyzer = PCAAnalyzer()
    X_pca, pca_reduced = pca_analyzer.get_pca_features(X, n_components=0.95)
    pca_feature_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]

    results_pca = classifier_analyzer.run_analysis(
        X_pca, Y, pca_feature_names, f"PCA Features ({X_pca.shape[1]} components)"
    )

    return results_original, results_pca


if __name__ == "__main__":
    main()
