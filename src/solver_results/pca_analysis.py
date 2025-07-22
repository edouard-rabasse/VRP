import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


class PCAAnalyzer:
    """Class to handle PCA analysis and visualization."""

    def __init__(self):
        self.pca = None
        self.scaler = None
        self.X_scaled = None
        self.X_pca = None

    def fit_transform(self, X):
        """Fit PCA and transform data."""
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        # Perform PCA
        self.pca = PCA()
        self.X_pca = self.pca.fit_transform(self.X_scaled)

        return self.X_pca

    def plot_analysis(self, X, Y, feature_names):
        """Perform comprehensive PCA analysis and visualization."""
        if self.pca is None:
            self.fit_transform(X)

        # Create subplots for analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Explained variance ratio
        self._plot_explained_variance(axes[0, 0])

        # 2. Cumulative explained variance
        self._plot_cumulative_variance(axes[0, 1])

        # 3. Feature contributions to first 2 PCs
        self._plot_feature_contributions(axes[1, 0], feature_names)

        # 4. 2D scatter plot in PC space
        self._plot_pc_scatter(axes[1, 1], Y)

        plt.tight_layout()
        plt.show()

        # Print insights
        self._print_insights(feature_names)

        return self.pca, self.scaler

    def _plot_explained_variance(self, ax):
        """Plot explained variance ratio."""
        ax.bar(
            range(1, len(self.pca.explained_variance_ratio_) + 1),
            self.pca.explained_variance_ratio_,
        )
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA Explained Variance")

        # Add cumulative variance line
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        ax.plot(range(1, len(cumsum) + 1), cumsum, "ro-", alpha=0.7)

    def _plot_cumulative_variance(self, ax):
        """Plot cumulative explained variance."""
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        ax.plot(range(1, len(cumsum) + 1), cumsum, "bo-")
        ax.axhline(y=0.95, color="r", linestyle="--", label="95% Variance")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("Cumulative Explained Variance")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_feature_contributions(self, ax, feature_names):
        """Plot feature contributions to first 2 PCs."""
        components_df = pd.DataFrame(
            self.pca.components_[:2].T, columns=["PC1", "PC2"], index=feature_names
        )

        sns.heatmap(
            components_df,
            annot=True,
            cmap="RdBu_r",
            center=0,
            ax=ax,
            cbar_kws={"label": "Component Loading"},
        )
        ax.set_title("Feature Contributions to PC1 & PC2")

    def _plot_pc_scatter(self, ax, Y):
        """Plot 2D scatter in PC space."""
        scatter = ax.scatter(
            self.X_pca[:, 0], self.X_pca[:, 1], c=Y, cmap="viridis", alpha=0.6
        )
        ax.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]:.2%} variance)")
        ax.set_title("Data in Principal Component Space")
        plt.colorbar(scatter, ax=ax, label="Target")

    def _print_insights(self, feature_names):
        """Print PCA insights."""
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)

        print("\n=== PCA Analysis Results ===")
        print(f"Total variance explained by first 2 components: {cumsum[1]:.2%}")
        if len(cumsum) > 2:
            print(f"Total variance explained by first 3 components: {cumsum[2]:.2%}")

        print("\nMost important features for PC1:")
        pc1_importance = (
            pd.Series(self.pca.components_[0], index=feature_names)
            .abs()
            .sort_values(ascending=False)
        )
        print(pc1_importance)

        print("\nMost important features for PC2:")
        pc2_importance = (
            pd.Series(self.pca.components_[1], index=feature_names)
            .abs()
            .sort_values(ascending=False)
        )
        print(pc2_importance)

    def get_pca_features(self, X, n_components=0.95):
        """Get PCA-transformed features."""
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        pca_reduced = PCA(n_components=n_components)
        X_pca = pca_reduced.fit_transform(X_scaled)

        return X_pca, pca_reduced

    def find_optimal_components(self, X, Y, max_components=None):
        """Find optimal number of PCA components for classification."""
        if self.X_scaled is None:
            self.fit_transform(X)

        max_components = max_components or min(X.shape[1], 10)
        results = []

        for n_comp in range(1, max_components + 1):
            pca_temp = PCA(n_components=n_comp)
            X_pca_temp = pca_temp.fit_transform(self.X_scaled)

            # Cross-validation
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            scores = cross_val_score(clf, X_pca_temp, Y, cv=5)

            results.append(
                {
                    "n_components": n_comp,
                    "mean_accuracy": scores.mean(),
                    "std_accuracy": scores.std(),
                    "explained_variance": pca_temp.explained_variance_ratio_.sum(),
                }
            )

        results_df = pd.DataFrame(results)

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.errorbar(
            results_df["n_components"],
            results_df["mean_accuracy"],
            yerr=results_df["std_accuracy"],
            marker="o",
        )
        ax1.set_xlabel("Number of Components")
        ax1.set_ylabel("Cross-Validation Accuracy")
        ax1.set_title("PCA Components vs Classification Performance")
        ax1.grid(True)

        ax2.plot(results_df["n_components"], results_df["explained_variance"], "ro-")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Explained Variance Ratio")
        ax2.set_title("PCA Components vs Explained Variance")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Find optimal
        optimal_idx = results_df["mean_accuracy"].idxmax()
        optimal_components = results_df.iloc[optimal_idx]["n_components"]

        print(f"Optimal number of components: {optimal_components}")
        print(f"Best accuracy: {results_df.iloc[optimal_idx]['mean_accuracy']:.3f}")

        return optimal_components, results_df
