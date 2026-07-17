"""Stage 1: Dataset profiling.

Produces a JSON-serializable report describing shape, types, missingness,
cardinality, distribution stats and correlations. This never mutates the
input dataframe - it is purely descriptive and feeds decisions made by the
downstream cleaning / outlier / feature-engineering stages.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class DataProfiler:
    def profile(self, df: pd.DataFrame) -> dict[str, Any]:
        n_rows, n_cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_cols = self._detect_datetime_columns(df)

        report: dict[str, Any] = {
            "n_rows": int(n_rows),
            "n_columns": int(n_cols),
            "n_duplicate_rows": int(df.duplicated().sum()),
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": [c for c in categorical_cols if c not in datetime_cols],
            "datetime_columns": datetime_cols,
            "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
            "missing_values": self._missing_report(df),
            "numeric_summary": self._numeric_summary(df, numeric_cols),
            "categorical_summary": self._categorical_summary(df, categorical_cols, datetime_cols),
            "high_cardinality_columns": [
                c for c in categorical_cols if c not in datetime_cols and df[c].nunique(dropna=True) > 50
            ],
            "correlations": self._correlation_report(df, numeric_cols),
        }
        return report

    @staticmethod
    def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
        detected = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().head(30)
            if sample.empty:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.9:
                    detected.append(col)
            except (ValueError, TypeError):
                continue
        return detected

    @staticmethod
    def _missing_report(df: pd.DataFrame) -> dict[str, dict[str, float]]:
        n = len(df)
        out = {}
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            out[col] = {
                "n_missing": n_missing,
                "pct_missing": round(n_missing / n, 4) if n else 0.0,
            }
        return out

    @staticmethod
    def _numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, dict[str, float]]:
        out = {}
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            out[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "skew": float(series.skew()) if len(series) > 2 else 0.0,
            }
        return out

    @staticmethod
    def _categorical_summary(
        df: pd.DataFrame, categorical_cols: list[str], datetime_cols: list[str]
    ) -> dict[str, dict[str, Any]]:
        out = {}
        for col in categorical_cols:
            if col in datetime_cols:
                continue
            series = df[col].dropna()
            n_unique = int(series.nunique())
            top_values = series.value_counts().head(5).to_dict()
            out[col] = {
                "n_unique": n_unique,
                "top_values": {str(k): int(v) for k, v in top_values.items()},
            }
        return out

    @staticmethod
    def _correlation_report(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
        if len(numeric_cols) < 2:
            return {}
        corr = df[numeric_cols].corr(numeric_only=True).round(3)
        # Flag strongly correlated pairs (potential redundant features)
        strong_pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                value = corr.iloc[i, j]
                if pd.notna(value) and abs(value) >= 0.85:
                    strong_pairs.append({"col_a": cols[i], "col_b": cols[j], "correlation": float(value)})
        return {
            "matrix": corr.to_dict(),
            "strong_pairs": strong_pairs,
        }
