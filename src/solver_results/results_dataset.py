import pandas as pd
import numpy as np
from typing import List, Optional
from .vrp_instance import (
    VRPInstance,
)  # Adjust this import path to your project structure


class VRPTableDataset:
    """
    A scikit-learn–friendly tabular dataset that merges all iterations from multiple VRPInstance objects.
    Each row corresponds to one IterationResult from one instance.
    """

    def __init__(
        self,
        vrp_instances: List[VRPInstance],
        feature_cols: Optional[List[str]] = None,
        target_col: str = "valid",
        dropna: bool = True,
    ):
        """
        Args:
            vrp_instances: List of VRPInstance objects.
            feature_cols: List of columns to use as features. If None, will auto-select all numeric ones except target.
            target_col: Name of the target column (e.g., 'valid', 'score', etc.).
            dropna: Whether to drop rows with missing feature/target values.
        """
        self.target_col = target_col

        # Flatten all iterations into a single DataFrame
        dfs = []
        for instance in vrp_instances:
            df = pd.DataFrame([it.to_dict() for it in instance.iterations])
            df["instance_number"] = instance.instance_number
            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)

        # Infer feature columns
        if feature_cols is None:
            feature_cols = (
                df_all.select_dtypes(include=["number"])
                .columns.drop(
                    [target_col, "instance_number", "iter", "time"], errors="ignore"
                )
                .tolist()
            )

        self.feature_cols = feature_cols

        if dropna:
            df_all = df_all.dropna(subset=feature_cols + [target_col])

        self.df = df_all
        self.X = df_all[feature_cols]
        self.y = df_all[target_col]
        self.instance_ids = df_all["instance_number"]

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the full DataFrame (features + target + instance_number)."""
        return self.df.copy()

    def get_sklearn_inputs(self):
        """Returns (X, y) as NumPy arrays for scikit-learn."""
        return self.X.to_numpy(), self.y.to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X.iloc[idx], self.y.iloc[idx]
