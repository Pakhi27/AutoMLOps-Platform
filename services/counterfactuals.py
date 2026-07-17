"""Counterfactual explanations — minimal actionable changes to flip prediction."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class CounterfactualExplainer:
    def explain(
        self,
        pipeline,
        entry: dict,
        record: dict[str, Any],
        *,
        max_changes: int = 3,
        steps: int = 8,
    ) -> dict[str, Any]:
        feature_columns = entry.get("feature_columns")
        if not feature_columns:
            raise ValueError(
                "Counterfactuals require a tabular model with feature_columns. "
                "Not supported for image, text, or time-series models."
            )
        task_type = entry.get("task_type", "classification")
        df = pd.DataFrame([record])
        for c in feature_columns:
            if c not in df.columns:
                df[c] = None
        df = df[feature_columns]

        original_pred = pipeline.predict(df)[0]
        if hasattr(pipeline, "predict_proba") and task_type == "classification":
            original_proba = pipeline.predict_proba(df)[0]
            target_class_idx = int(np.argmax(original_proba))
            original_conf = float(original_proba[target_class_idx])
        else:
            original_proba = None
            target_class_idx = None
            original_conf = None

        label_classes = entry.get("label_classes")
        if label_classes:
            pred_label = label_classes[int(original_pred)]
        else:
            pred_label = original_pred

        # Reference ranges from training data
        settings_path = entry.get("reference_data_path")
        ref_df = None
        if settings_path:
            try:
                from app.utils.io_utils import read_csv_safely
                ref_df = read_csv_safely(settings_path)
            except Exception:
                pass

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        counterfactuals: list[dict[str, Any]] = []

        for col in numeric_cols[:10]:
            if col not in df.columns:
                continue
            base_val = float(df[col].iloc[0]) if pd.notna(df[col].iloc[0]) else 0.0
            if ref_df is not None and col in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df[col]):
                lo, hi = float(ref_df[col].quantile(0.05)), float(ref_df[col].quantile(0.95))
            else:
                lo, hi = base_val * 0.5, base_val * 1.5 if base_val else (-1, 1)

            for direction, mult in (("increase", 1), ("decrease", -1)):
                for step in range(1, steps + 1):
                    delta = (hi - lo) * step / steps
                    new_val = base_val + mult * delta
                    trial = df.copy()
                    trial[col] = new_val
                    new_pred = pipeline.predict(trial)[0]
                    if new_pred != original_pred:
                        cf_label = label_classes[int(new_pred)] if label_classes else new_pred
                        counterfactuals.append({
                            "feature": col,
                            "direction": direction,
                            "original_value": round(base_val, 4),
                            "suggested_value": round(new_val, 4),
                            "change": round(new_val - base_val, 4),
                            "new_prediction": cf_label,
                        })
                        break
            if len(counterfactuals) >= max_changes:
                break

        return {
            "original_prediction": pred_label,
            "original_confidence": round(original_conf, 4) if original_conf is not None else None,
            "task_type": task_type,
            "counterfactuals": counterfactuals[:max_changes],
            "summary": _summary(pred_label, counterfactuals[:max_changes]),
        }


def _summary(prediction: Any, cfs: list[dict]) -> str:
    if not cfs:
        return f"Prediction: {prediction}. No simple single-feature counterfactual found in search range."
    parts = []
    for cf in cfs:
        parts.append(
            f"{cf['direction'].capitalize()} `{cf['feature']}` by {cf['change']:+.4f} → {cf['new_prediction']}"
        )
    return f"Prediction: **{prediction}**. To change: " + " OR ".join(parts)
