"""Stage 4: Feature engineering with datetime expansion, encoding, interactions, and binning."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        datetime_columns: list[str] | None = None,
        skew_threshold: float = 1.0,
        onehot_max_cardinality: int = 10,
        enable_interactions: bool = True,
        max_interactions: int = 3,
        enable_binning: bool = True,
        binning_max_unique: int = 20,
    ) -> None:
        self.datetime_columns = datetime_columns or []
        self.skew_threshold = skew_threshold
        self.onehot_max_cardinality = onehot_max_cardinality
        self.enable_interactions = enable_interactions
        self.max_interactions = max_interactions
        self.enable_binning = enable_binning
        self.binning_max_unique = binning_max_unique

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineerTransformer":
        X = X.copy()
        self.report_: dict[str, Any] = {
            "datetime_expanded": [],
            "log_transformed": [],
            "onehot_encoded": {},
            "frequency_encoded": [],
            "binned_columns": [],
            "interaction_pairs": [],
        }

        self.datetime_columns_ = [c for c in self.datetime_columns if c in X.columns]
        X = self._expand_datetime(X, fitting=True)

        remaining_numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        self.log_transform_cols_: list[str] = []
        for col in remaining_numeric:
            series = X[col].dropna()
            if series.empty or (series < 0).any():
                continue
            skew = series.skew()
            if pd.notna(skew) and abs(skew) > self.skew_threshold:
                self.log_transform_cols_.append(col)
        self.report_["log_transformed"] = self.log_transform_cols_

        self.bin_cols_: list[str] = []
        self.bin_edges_: dict[str, np.ndarray] = {}
        if self.enable_binning:
            for col in remaining_numeric:
                if col in self.log_transform_cols_:
                    continue
                n_unique = X[col].nunique(dropna=True)
                if n_unique > self.binning_max_unique:
                    self.bin_cols_.append(col)
                    series = X[col].dropna()
                    if len(series) >= 2:
                        try:
                            _, edges = pd.qcut(series, q=5, retbins=True, duplicates="drop")
                            self.bin_edges_[col] = edges
                        except ValueError:
                            pass
            self.report_["binned_columns"] = self.bin_cols_

        self.interaction_pairs_: list[tuple[str, str]] = []
        if self.enable_interactions and y is not None:
            numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c in X.columns]
            if len(numeric_cols) >= 2:
                y_series = pd.Series(y, index=X.index)
                correlations: list[tuple[float, str, str]] = []
                for i, col_a in enumerate(numeric_cols):
                    for col_b in numeric_cols[i + 1 :]:
                        try:
                            corr = abs(X[col_a].corr(y_series))
                        except Exception:
                            corr = 0.0
                        correlations.append((corr if pd.notna(corr) else 0.0, col_a, col_b))
                correlations.sort(reverse=True)
                self.interaction_pairs_ = [
                    (a, b) for _, a, b in correlations[: self.max_interactions]
                ]
            self.report_["interaction_pairs"] = [list(p) for p in self.interaction_pairs_]

        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.onehot_categories_: dict[str, list[str]] = {}
        self.freq_maps_: dict[str, dict[str, float]] = {}

        for col in categorical_cols:
            n_unique = X[col].nunique(dropna=True)
            if n_unique <= self.onehot_max_cardinality:
                categories = sorted(X[col].dropna().astype(str).unique().tolist())
                self.onehot_categories_[col] = categories
                self.report_["onehot_encoded"][col] = categories
            else:
                counts = X[col].astype(str).value_counts(normalize=True)
                self.freq_maps_[col] = counts.to_dict()
                self.report_["frequency_encoded"].append(col)

        transformed = self._apply(X)
        self.output_columns_ = transformed.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = self._expand_datetime(X, fitting=False)
        X = self._apply(X)
        for col in self.output_columns_:
            if col not in X.columns:
                X[col] = 0
        X = X[self.output_columns_]
        return X

    def _expand_datetime(self, X: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        cols = self.datetime_columns_ if not fitting else [c for c in self.datetime_columns if c in X.columns]
        for col in cols:
            if col not in X.columns:
                continue
            parsed = pd.to_datetime(X[col], errors="coerce", format="mixed")
            X[f"{col}_year"] = parsed.dt.year
            X[f"{col}_month"] = parsed.dt.month
            X[f"{col}_day"] = parsed.dt.day
            X[f"{col}_dayofweek"] = parsed.dt.dayofweek
            X[f"{col}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
            X = X.drop(columns=[col])
            if fitting:
                self.report_["datetime_expanded"].append(col)
        return X

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.log_transform_cols_:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))

        for col in self.bin_cols_:
            if col not in X.columns:
                continue
            if col in self.bin_edges_:
                binned = pd.cut(
                    X[col], bins=self.bin_edges_[col], labels=False, include_lowest=True
                )
                X[f"{col}__binned"] = binned.fillna(0).astype(int)
            else:
                X[f"{col}__binned"] = 0

        for col_a, col_b in self.interaction_pairs_:
            if col_a in X.columns and col_b in X.columns:
                X[f"{col_a}__x__{col_b}"] = X[col_a] * X[col_b]

        for col, categories in self.onehot_categories_.items():
            if col not in X.columns:
                continue
            str_col = X[col].astype(str)
            for cat in categories:
                X[f"{col}__{cat}"] = (str_col == cat).astype(int)
            X = X.drop(columns=[col])

        for col, freq_map in self.freq_maps_.items():
            if col not in X.columns:
                continue
            X[f"{col}__freq"] = X[col].astype(str).map(freq_map).fillna(0.0)
            X = X.drop(columns=[col])

        non_numeric = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        if non_numeric:
            X = X.drop(columns=non_numeric)

        # GradientBoosting and similar models reject NaN — fill any leftovers.
        return X.fillna(0)

    def get_report(self) -> dict[str, Any]:
        return getattr(self, "report_", {})
