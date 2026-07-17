"""Feature importance extraction from trained pipelines (SHAP or model-native)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.services.explainability import ExplainabilityService
from app.services.model_registry import get_model_registry
from app.utils.io_utils import read_csv_safely


def _get_model_step(pipeline) -> Any:
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("model", pipeline.steps[-1][1])
    return pipeline


def extract_from_pipeline(pipeline, X_background: pd.DataFrame, task_type: str) -> list[dict[str, Any]]:
    """Compute mean |SHAP| importance over raw input columns."""
    try:
        if task_type == "classification" and hasattr(pipeline, "predict_proba"):
            predict_fn = lambda data: pipeline.predict_proba(data)
        else:
            predict_fn = lambda data: pipeline.predict(data)

        explainer = ExplainabilityService(sample_size=min(100, len(X_background)))
        explainer.build_explainer(predict_fn, X_background)
        sample = X_background.sample(min(50, len(X_background)), random_state=42)
        encoded = explainer._encode(sample, fit=False)
        shap_values = explainer._explainer(encoded)
        values = np.array(shap_values.values)
        if values.ndim == 3:
            values = np.abs(values).mean(axis=2)
        else:
            values = np.abs(values)
        mean_abs = values.mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        return [
            {
                "feature": explainer._column_order[i],
                "importance": round(float(mean_abs[i]), 6),
                "rank": rank + 1,
            }
            for rank, i in enumerate(order)
        ]
    except Exception:
        pass

    model = _get_model_step(pipeline)
    inner = model
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("model", model.steps[-1][1])

    if hasattr(inner, "feature_importances_"):
        names = X_background.columns.tolist()
        importances = inner.feature_importances_
        if len(importances) != len(names):
            names = [f"feature_{i}" for i in range(len(importances))]
        pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return [
            {"feature": n, "importance": round(float(v), 6), "rank": i + 1}
            for i, (n, v) in enumerate(pairs)
        ]

    return [{"feature": c, "importance": 0.0, "rank": i + 1} for i, c in enumerate(X_background.columns)]


def load_or_compute(job_id: str) -> list[dict[str, Any]]:
    from app.core.config import get_settings

    settings = get_settings()
    cache_path = settings.artifacts_dir / job_id / "feature_importance.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    registry = get_model_registry()
    entry = registry.get(job_id)
    if entry is None:
        raise FileNotFoundError(f"No model for job '{job_id}'.")

    pipeline = joblib.load(entry["pipeline_path"])
    ref_path = settings.reference_dir / f"{job_id}.csv"
    if not ref_path.exists():
        raise FileNotFoundError("Reference data missing for feature importance.")
    X_ref = read_csv_safely(ref_path)
    importances = extract_from_pipeline(pipeline, X_ref, entry.get("task_type", "classification"))
    cache_path.write_text(json.dumps(importances, indent=2), encoding="utf-8")
    return importances
