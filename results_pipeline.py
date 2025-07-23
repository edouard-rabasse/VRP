from src.solver_results.results_dataset import VRPTableDataset
from src.solver_results.vrp_instance import VRPInstance
from src.solver_results.pca_analysis import PCAAnalyzer
from src.solver_results.easy_classifier import classify_vrp_instance
from sklearn import tree
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


def main():
    """Main pipeline function."""
    path = "output/resnet_1_1_2e-07_version2"

    test_enhanced_classifier(path)


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


# Example usage in your pipeline
from src.solver_results.easy_classifier import (
    EnhancedVRPClassifier,
    ClassificationCriteria,
    classify_vrp_instance,
    classify_vrp_instance_multi,
)


def test_enhanced_classifier(path):
    # Load your instances
    vrp_instances = load_vrp_instances(path, range(1001, 1076))

    classifier = EnhancedVRPClassifier()

    # Test different approaches
    approaches = {
        "aggressive": "aggressive",
    }

    results = {}

    for approach_name, preset in approaches.items():
        print(f"\n=== {approach_name.upper()} APPROACH ===")

        correct = 0
        first_time = 0
        total = 0

        for instance in vrp_instances:
            result = classifier.classify_with_preset(instance, preset)
            total += 1

            if result.valid:
                correct += 1
            if result.first_time_valid:
                first_time += 1

            # Print details for first few instances
            if total <= 3:
                print(f"Instance {instance.instance_number}:")
                print(f"  Result: {result.reason}")
                print(f"  Satisfied criteria: {result.satisfied_criteria}")

        results[approach_name] = {
            "total": total,
            "correct": correct,
            "first_time": first_time,
            "success_rate": correct / total if total > 0 else 0,
        }

        print(f"Success rate: {results[approach_name]['success_rate']:.2%}")
    return results


# Custom criteria example
def test_custom_criteria():
    custom_criteria = ClassificationCriteria(
        classifier_score_threshold=0.4,
        entropy_score_threshold=0.25,
        max_violations=3,
        entropy_trend_threshold=-0.05,  # Entropy must be decreasing
        require_valid=True,
        combination_mode="WEIGHTED",
        weights={
            "classifier_score": 0.3,
            "entropy_score": 0.3,
            "valid": 0.2,
            "entropy_trend": 0.2,
        },
    )

    classifier = EnhancedVRPClassifier(custom_criteria)

    # Test on instances
    for instance in vrp_instances[:5]:
        result = classifier.classify(instance)
        print(f"Instance {instance.instance_number}: {result.reason}")
        print(
            f"  Score: {result.classification_score:.3f}, Confidence: {result.confidence:.3f}"
        )


if __name__ == "__main__":
    main()
