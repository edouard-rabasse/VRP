import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


class ClassificationAnalyzer:
    """Class to handle classification analysis and comparison."""

    def __init__(self):
        self.classifiers = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            "SVM": SVC(
                kernel="rbf", probability=True, random_state=42, class_weight="balanced"
            ),
            "Logistic Regression": LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            ),
            "Decision Tree": tree.DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
            ),
        }

    def run_analysis(self, X, Y, feature_names, title):
        """Run comprehensive classification analysis."""
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.5, random_state=42, stratify=Y
        )

        results = {}

        print(f"\n{title}:")
        print("=" * 60)

        for clf_name, clf in self.classifiers.items():
            result = self._evaluate_classifier(
                clf, clf_name, X_train, X_test, Y_train, Y_test
            )
            results[clf_name] = result

        # Find best classifier
        best_clf_name = max(results.keys(), key=lambda k: results[k]["cv_mean"])
        print(f"\nBest Classifier: {best_clf_name}")

        # Detailed analysis
        self.analyze_best_classifier(
            results[best_clf_name], best_clf_name, feature_names, title, Y_test
        )
        self.compare_classifiers(results, title)

        return results

    def _evaluate_classifier(self, clf, clf_name, X_train, X_test, Y_train, Y_test):
        """Evaluate a single classifier."""
        # Fit and predict
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = (
            clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        )

        # Calculate metrics
        cm = confusion_matrix(Y_test, y_pred)
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

        # Cross-validation score
        cv_scores = cross_val_score(clf, X_train, Y_train, cv=5)

        # ROC AUC if probabilities available
        roc_auc = (
            roc_auc_score(Y_test, y_pred_proba) if y_pred_proba is not None else None
        )

        result = {
            "accuracy": accuracy,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": cm,
            "classifier": clf,
            "roc_auc": roc_auc,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
            "Y_test": Y_test,
        }

        # Print results
        print(f"\n{clf_name}:")
        print(f"  Test Accuracy: {accuracy:.3f}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        if roc_auc:
            print(f"  ROC AUC: {roc_auc:.3f}")
        print(f"  Confusion Matrix:\n{cm}")

        return result

    def analyze_best_classifier(
        self, best_result, clf_name, feature_names, title, Y_test=None
    ):
        """Analyze the best performing classifier in detail."""
        plt.figure(figsize=(12, 9))

        # Plot 1: Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = best_result["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Invalid", "Valid"],
            yticklabels=["Invalid", "Valid"],
        )
        plt.title(f"Confusion Matrix - {clf_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # Plot 2: Feature Importance
        if hasattr(best_result["classifier"], "feature_importances_"):
            plt.subplot(2, 3, 2)
            importances = best_result["classifier"].feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.bar(range(len(importances)), importances[indices])
            plt.title(f"Feature Importance - {clf_name}")
            plt.xticks(
                range(len(importances)),
                [feature_names[i] for i in indices],
                rotation=45,
            )

            print(f"\nFeature Importance ({clf_name}):")
            for i in indices:
                print(f"  {feature_names[i]}: {importances[i]:.4f}")

        # Plot 3: ROC Curve
        if best_result["probabilities"] is not None and Y_test is not None:
            plt.subplot(2, 3, 3)
            fpr, tpr, _ = roc_curve(Y_test, best_result["probabilities"])
            plt.plot(fpr, tpr, label=f'ROC AUC = {best_result["roc_auc"]:.3f}')
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()

        # Plot 4: Decision Tree visualization
        if isinstance(best_result["classifier"], tree.DecisionTreeClassifier):
            plt.subplot(2, 3, (4, 6))
            tree.plot_tree(
                best_result["classifier"],
                filled=True,
                feature_names=feature_names,
                class_names=["Invalid", "Valid"],
                fontsize=8,
                max_depth=3,
            )
            plt.title(f"Decision Tree - {clf_name}")

        plt.suptitle(f"Best Classifier Analysis - {title}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def compare_classifiers(self, results, title):
        """Compare all classifiers performance."""
        comparison_data = []
        for clf_name, result in results.items():
            comparison_data.append(
                {
                    "Classifier": clf_name,
                    "Test Accuracy": result["accuracy"],
                    "CV Mean": result["cv_mean"],
                    "CV Std": result["cv_std"],
                    "ROC AUC": result["roc_auc"] if result["roc_auc"] else "N/A",
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        # Plot comparison
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        x_pos = np.arange(len(comparison_df))
        plt.bar(x_pos, comparison_df["Test Accuracy"], alpha=0.7, label="Test Accuracy")
        plt.bar(x_pos, comparison_df["CV Mean"], alpha=0.7, label="CV Mean")
        plt.xlabel("Classifier")
        plt.ylabel("Accuracy")
        plt.title(f"Classifier Comparison - {title}")
        plt.xticks(x_pos, comparison_df["Classifier"], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        roc_scores = [r for r in comparison_df["ROC AUC"] if r != "N/A"]
        roc_names = [
            comparison_df.iloc[i]["Classifier"]
            for i, r in enumerate(comparison_df["ROC AUC"])
            if r != "N/A"
        ]

        if roc_scores:
            plt.bar(range(len(roc_scores)), roc_scores, alpha=0.7)
            plt.xlabel("Classifier")
            plt.ylabel("ROC AUC")
            plt.title("ROC AUC Comparison")
            plt.xticks(range(len(roc_names)), roc_names, rotation=45)
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print comparison table
        print(f"\nClassifier Comparison - {title}:")
        print(comparison_df.to_string(index=False, float_format="%.3f"))
