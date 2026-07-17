"""Stage 3: Outlier detection & handling.

Two strategies are supported:
  - "iqr": per-column Tukey fences (Q1 - k*IQR, Q3 + k*IQR), fit on train data.
  - "isolation_forest": multivariate anomaly detection.

Bounds/model are fit once on training data and reused at transform time so
behaviour is identical between training and inference (row-count preserving:
outliers are capped/clipped, never dropped, so this is safe inside a Pipeline).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


class OutlierCapTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        numeric_columns: list[str] | None = None,
        method: str = "iqr",
        iqr_multiplier: float = 1.5,
        random_state: int = 42,
    ) -> None:
        self.numeric_columns = numeric_columns
        self.method = method
        self.iqr_multiplier = iqr_multiplier
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OutlierCapTransformer":
        cols = self.numeric_columns or X.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_columns_ = cols
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.report_: dict[str, Any] = {"method": self.method, "columns_processed": cols, "n_outliers_per_column": {}}

        for col in cols:
            series = X[col].dropna()
            if series.empty:
                self.bounds_[col] = (-np.inf, np.inf)
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            self.bounds_[col] = (float(lower), float(upper))
            n_out = int(((series < lower) | (series > upper)).sum())
            self.report_["n_outliers_per_column"][col] = n_out

        if self.method == "isolation_forest" and cols:
            fill_values = X[cols].median(numeric_only=True)
            self.iso_forest_ = IsolationForest(
                n_estimators=200, contamination="auto", random_state=self.random_state
            )
            self.iso_forest_.fit(X[cols].fillna(fill_values))
            self._iso_fill_values_ = fill_values
        else:
            self.iso_forest_ = None

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)

        if self.method == "isolation_forest" and self.iso_forest_ is not None:
            cols = [c for c in self.numeric_columns_ if c in X.columns]
            if cols:
                filled = X[cols].fillna(self._iso_fill_values_)
                scores = self.iso_forest_.decision_function(filled)
                X["_anomaly_score"] = scores

        return X

    def get_report(self) -> dict[str, Any]:
        return getattr(self, "report_", {})
