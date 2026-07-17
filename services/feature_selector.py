"""Automatic feature selection after feature engineering."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep only selected columns through the sklearn pipeline."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns_ = list(X.columns) if hasattr(X, "columns") else list(range(X.shape[1]))
        else:
            cols = list(X.columns) if hasattr(X, "columns") else self.columns
            self.columns_ = [c for c in self.columns if c in cols]
        return self

    def transform(self, X):
        if hasattr(X, "columns"):
            return X[self.columns_]
        return X


class FeatureSelector:
    def __init__(
        self,
        method: str = "mutual_info",
        max_features: int | None = None,
        variance_threshold: float = 0.0,
    ):
        self.method = method
        self.max_features = max_features
        self.variance_threshold = variance_threshold

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        original_cols = list(X.columns)
        n_original = len(original_cols)

        # Drop near-zero variance
        variances = X.var(numeric_only=True)
        low_var = [c for c in variances.index if variances[c] <= self.variance_threshold]
        X_work = X.drop(columns=low_var, errors="ignore")

        k = self.max_features
        if k is None:
            k = max(5, min(50, int(len(X_work.columns) * 0.6)))
        k = min(k, len(X_work.columns))

        if k >= len(X_work.columns):
            return X, {
                "method": self.method,
                "original_count": n_original,
                "selected_count": n_original,
                "removed_count": 0,
                "selected_features": original_cols,
                "removed_features": [],
                "scores": {},
            }

        X_num = X_work.select_dtypes(include=[np.number])
        if X_num.empty or len(X_num.columns) == 0:
            return X, {
                "method": self.method,
                "original_count": n_original,
                "selected_count": n_original,
                "removed_count": 0,
                "selected_features": original_cols,
                "removed_features": [],
                "scores": {},
            }

        non_num = [c for c in X_work.columns if c not in X_num.columns]
        X_fill = X_num.fillna(X_num.median())

        if self.method == "mutual_info":
            if task_type == "classification":
                scores = mutual_info_classif(X_fill, y, random_state=42)
            else:
                scores = mutual_info_regression(X_fill, y, random_state=42)
        else:
            if task_type == "classification":
                scores, _ = f_classif(X_fill, y)
            else:
                scores, _ = f_regression(X_fill, y)

        score_map = {col: float(scores[i]) for i, col in enumerate(X_num.columns)}
        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        selected_num = [c for c, _ in ranked[:k]]
        selected = selected_num + non_num
        removed = [c for c in original_cols if c not in selected]

        X_selected = X[selected].copy()
        return X_selected, {
            "method": self.method,
            "original_count": n_original,
            "selected_count": len(selected),
            "removed_count": len(removed),
            "selected_features": selected,
            "removed_features": removed,
            "scores": dict(ranked[:20]),
        }
