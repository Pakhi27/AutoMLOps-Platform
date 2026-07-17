"""Exploratory data analysis: univariate stats, target analysis, feature-vs-target relationships."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.services.model_selector import ModelSelector


class EDAService:
    def analyze(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        y = df[target_column]
        X = df.drop(columns=[target_column])
        selector = ModelSelector()
        task_type = selector.detect_task_type(y)

        result = {
            "target_column": target_column,
            "task_type": task_type,
            "n_rows": len(df),
            "target_analysis": self._target_analysis(y, task_type),
            "numeric_features": self._numeric_vs_target(X, y, task_type),
            "categorical_features": self._categorical_vs_target(X, y, task_type),
            "missing_by_column": {
                c: round(float(df[c].isna().mean()), 4) for c in df.columns
            },
            "correlation_with_target": self._target_correlations(X, y, task_type),
        }
        return self._sanitize(result)

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: EDAService._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [EDAService._sanitize(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return 0.0
        if isinstance(obj, (np.floating, np.integer)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return val
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    def _target_analysis(self, y: pd.Series, task_type: str) -> dict[str, Any]:
        if task_type == "classification":
            counts = y.value_counts(dropna=False)
            total = len(y)
            distribution = {
                str(k): {"count": int(v), "pct": round(v / total, 4)} for k, v in counts.items()
            }
            majority = counts.max() / total if total else 0
            return {
                "type": "classification",
                "n_classes": int(y.nunique()),
                "distribution": distribution,
                "imbalance_ratio": round(float(majority), 4),
                "is_imbalanced": majority > 0.7,
            }

        series = pd.to_numeric(y, errors="coerce").dropna()
        hist, edges = np.histogram(series, bins=min(20, max(5, series.nunique())))
        return {
            "type": "regression",
            "mean": float(series.mean()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "histogram": {
                "counts": hist.tolist(),
                "bin_edges": [round(float(e), 4) for e in edges],
            },
        }

    def _numeric_vs_target(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = X[col].dropna()
            if series.empty:
                continue
            hist, edges = np.histogram(series, bins=min(15, max(5, series.nunique())))
            entry: dict[str, Any] = {
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "skew": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
                "missing_pct": round(float(X[col].isna().mean()), 4),
                "histogram": {
                    "counts": hist.tolist(),
                    "bin_edges": [round(float(e), 4) for e in edges],
                },
            }
            if task_type == "classification":
                grouped = (
                    pd.DataFrame({col: X[col], "target": y})
                    .groupby("target")[col]
                    .mean()
                    .round(4)
                )
                entry["mean_by_class"] = {str(k): float(v) for k, v in grouped.items()}
            else:
                try:
                    corr = float(pd.to_numeric(X[col], errors="coerce").corr(pd.to_numeric(y, errors="coerce")))
                    entry["correlation_with_target"] = round(corr, 4) if pd.notna(corr) else 0.0
                except Exception:
                    entry["correlation_with_target"] = 0.0
            out[col] = entry
        return out

    def _categorical_vs_target(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        for col in cat_cols:
            series = X[col].astype(str)
            value_counts = series.value_counts().head(10)
            entry: dict[str, Any] = {
                "n_unique": int(series.nunique()),
                "top_values": {str(k): int(v) for k, v in value_counts.items()},
                "missing_pct": round(float(X[col].isna().mean()), 4),
            }
            if task_type == "classification":
                ct = pd.crosstab(series, y, normalize="index").round(4)
                entry["target_rate_by_category"] = {
                    str(idx): {str(c): float(v) for c, v in row.items()} for idx, row in ct.iterrows()
                }
            else:
                grouped = pd.DataFrame({col: series, "target": pd.to_numeric(y, errors="coerce")}).groupby(col)[
                    "target"
                ].mean()
                entry["mean_target_by_category"] = {
                    str(k): round(float(v), 4) for k, v in grouped.head(10).items()
                }
            out[col] = entry
        return out

    def _target_correlations(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if task_type == "classification":
            y_num = pd.Series(pd.Categorical(y).codes, index=y.index)
        else:
            y_num = pd.to_numeric(y, errors="coerce")

        for col in X.select_dtypes(include=[np.number]).columns:
            try:
                corr = float(X[col].corr(y_num))
                if pd.notna(corr):
                    results.append({"feature": col, "correlation": round(corr, 4)})
            except Exception:
                continue

        results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return results[:15]
