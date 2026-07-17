"""Stage 8: Model explainability with SHAP.

The explainer wraps the *entire fitted pipeline* (preprocessing + model) as a
black-box callable. This is slightly slower than a model-specific
TreeExplainer, but it means SHAP attributions are reported against the
original, human-readable input columns (e.g. "age", "city") rather than the
one-hot-expanded internal feature names - much more useful for an end user
calling the /predict API.

SHAP's maskers operate on purely numeric arrays (they run `numpy.isfinite`
checks internally), so raw string/categorical/ID columns are encoded to
integer codes before being handed to SHAP, and decoded back to their
original values just before calling the real pipeline's predict function.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # headless rendering, required in API/container context
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class ExplainabilityService:
    def __init__(self, sample_size: int = 200):
        self.sample_size = sample_size
        self._explainer: shap.Explainer | None = None
        self._raw_predict_fn: Callable[[pd.DataFrame], np.ndarray] | None = None
        self._column_order: list[str] = []
        self._categories: dict[str, list[str]] = {}

    def build_explainer(self, predict_fn: Callable[[pd.DataFrame], np.ndarray], X_background: pd.DataFrame) -> shap.Explainer:
        background = X_background.sample(min(len(X_background), self.sample_size), random_state=42)
        self._raw_predict_fn = predict_fn
        self._column_order = background.columns.tolist()

        encoded_background = self._encode(background, fit=True)
        masker = shap.maskers.Independent(encoded_background, max_samples=min(len(encoded_background), 100))
        self._explainer = shap.Explainer(self._wrapped_predict, masker, feature_names=self._column_order)
        return self._explainer

    def _encode(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = X.reindex(columns=self._column_order)
        encoded = pd.DataFrame(index=X.index)
        for col in self._column_order:
            series = X[col]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                encoded[col] = series.astype(float)
                continue
            str_series = series.astype(str).fillna("__missing__")
            if fit:
                self._categories[col] = sorted(str_series.unique().tolist())
            categories = self._categories.get(col, [])
            cat_to_code = {c: float(i) for i, c in enumerate(categories)}
            encoded[col] = str_series.map(cat_to_code).fillna(-1.0)
        return encoded

    def _decode(self, X_encoded: np.ndarray) -> pd.DataFrame:
        arr = np.atleast_2d(X_encoded)
        df = pd.DataFrame(arr, columns=self._column_order)
        for col, categories in self._categories.items():
            if not categories:
                continue
            idx = df[col].round().clip(0, len(categories) - 1).astype(int)
            df[col] = [categories[i] for i in idx]
        return df

    def _wrapped_predict(self, X_encoded: np.ndarray) -> np.ndarray:
        df = self._decode(X_encoded)
        return self._raw_predict_fn(df)

    def summary_plot(self, X_sample: pd.DataFrame, output_path: str | Path) -> str:
        """Renders a mean(|SHAP value|) feature-importance bar chart.

        Built manually (rather than via `shap.summary_plot`) because that
        helper's automatic handling of multi-output (predict_proba) arrays is
        version-fragile - it can misinterpret a (n_samples, n_features,
        n_classes) array as pairwise interaction values. Averaging the
        absolute SHAP value across classes ourselves is simple and robust.
        """
        if self._explainer is None:
            raise RuntimeError("Explainer has not been built yet - call build_explainer first.")
        sample = X_sample.sample(min(len(X_sample), self.sample_size), random_state=42)
        encoded_sample = self._encode(sample, fit=False)
        shap_values = self._explainer(encoded_sample)

        values = np.array(shap_values.values)
        if values.ndim == 3:  # (n_samples, n_features, n_classes)
            values = np.abs(values).mean(axis=2)
        else:
            values = np.abs(values)

        mean_abs_importance = values.mean(axis=0)
        order = np.argsort(mean_abs_importance)[::-1]
        features_sorted = [self._column_order[i] for i in order]
        importances_sorted = mean_abs_importance[order]

        fig_height = max(4, 0.35 * len(features_sorted))
        plt.figure(figsize=(8, fig_height))
        y_pos = np.arange(len(features_sorted))
        plt.barh(y_pos, importances_sorted, color="#4C72B0")
        plt.yticks(y_pos, features_sorted)
        plt.gca().invert_yaxis()
        plt.xlabel("mean(|SHAP value|)")
        plt.title("Global feature importance (SHAP)")
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120)
        plt.close()
        return str(output_path)

    def explain_rows(self, X_rows: pd.DataFrame) -> list[dict[str, float]]:
        if self._explainer is None:
            raise RuntimeError("Explainer has not been built yet - call build_explainer first.")
        encoded_rows = self._encode(X_rows, fit=False)
        shap_values = self._explainer(encoded_rows)
        values = np.array(shap_values.values)
        if values.ndim == 3:  # multi-class: (n_rows, n_features, n_classes) -> mean abs impact
            values = np.abs(values).mean(axis=2)
        results = []
        for row in values:
            results.append({self._column_order[i]: float(row[i]) for i in range(len(self._column_order))})
        return results
