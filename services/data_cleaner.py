"""Stage 2: Missing-value cleaning.

Implemented as an sklearn-compatible transformer (fit on train, applied to
train/test/inference alike) so the exact same cleaning logic - including
which columns get dropped and which impute values are used - is replayed
consistently at prediction time. The fitted transformer is bundled into the
final sklearn Pipeline and persisted with the model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        missing_drop_threshold: float = 0.9,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "most_frequent",
    ) -> None:
        self.missing_drop_threshold = missing_drop_threshold
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DataCleanerTransformer":
        X = X.copy()
        n = len(X)

        self.columns_to_drop_: list[str] = []
        missing_pct = X.isna().mean()
        self.columns_to_drop_.extend(missing_pct[missing_pct > self.missing_drop_threshold].index.tolist())

        constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        self.columns_to_drop_.extend([c for c in constant_cols if c not in self.columns_to_drop_])

        remaining = [c for c in X.columns if c not in self.columns_to_drop_]
        self.numeric_cols_ = [c for c in remaining if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols_ = [c for c in remaining if c not in self.numeric_cols_]

        self.impute_values_: dict[str, Any] = {}
        for col in self.numeric_cols_:
            series = X[col].dropna()
            if series.empty:
                self.impute_values_[col] = 0.0
            elif self.numeric_impute_strategy == "mean":
                self.impute_values_[col] = float(series.mean())
            else:
                self.impute_values_[col] = float(series.median())

        for col in self.categorical_cols_:
            series = X[col].dropna()
            if series.empty:
                self.impute_values_[col] = "missing"
            elif self.categorical_impute_strategy == "most_frequent":
                self.impute_values_[col] = series.mode().iloc[0]
            else:
                self.impute_values_[col] = "missing"

        self.n_rows_seen_ = n
        self.report_: dict[str, Any] = {
            "dropped_columns": self.columns_to_drop_,
            "imputed_columns": list(self.impute_values_.keys()),
            "impute_values": {k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in self.impute_values_.items()},
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # NOTE: row-count must be preserved here (this runs inside an sklearn
        # Pipeline, which does not re-sample `y` if a transformer changes the
        # number of rows). Duplicate-row removal is handled once, upfront, on
        # the raw training dataframe in the orchestrator - not here.
        X = X.copy()
        X = X.drop(columns=[c for c in self.columns_to_drop_ if c in X.columns], errors="ignore")

        for col, value in self.impute_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)

        return X

    def get_report(self) -> dict[str, Any]:
        return getattr(self, "report_", {})
