"""Active learning — flag low-confidence predictions for human review."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ActiveLearningService:
    def __init__(self, uncertainty_threshold: float = 0.55):
        self.uncertainty_threshold = uncertainty_threshold

    def score_batch(
        self,
        pipeline,
        entry: dict,
        df: pd.DataFrame,
    ) -> dict[str, Any]:
        feature_columns = entry["feature_columns"]
        task_type = entry.get("task_type", "classification")
        for c in feature_columns:
            if c not in df.columns:
                df[c] = None
        df = df[feature_columns].copy()

        preds = pipeline.predict(df)
        rows: list[dict[str, Any]] = []

        if hasattr(pipeline, "predict_proba") and task_type == "classification":
            probas = pipeline.predict_proba(df)
            label_classes = entry.get("label_classes")
            for i in range(len(df)):
                proba = probas[i]
                conf = float(np.max(proba))
                entropy = float(-np.sum(proba * np.log(proba + 1e-12)))
                uncertain = conf < self.uncertainty_threshold
                pred_label = label_classes[int(preds[i])] if label_classes else int(preds[i])
                rows.append({
                    "row_index": i,
                    "prediction": pred_label,
                    "confidence": round(conf, 4),
                    "entropy": round(entropy, 4),
                    "needs_review": uncertain,
                    "action": "Add to retraining queue" if uncertain else "Auto-accept",
                })
        else:
            for i in range(len(df)):
                rows.append({
                    "row_index": i,
                    "prediction": float(preds[i]),
                    "confidence": None,
                    "entropy": None,
                    "needs_review": False,
                    "action": "Auto-accept",
                })

        review_queue = [r for r in rows if r["needs_review"]]
        return {
            "n_rows": len(rows),
            "n_needs_review": len(review_queue),
            "uncertainty_threshold": self.uncertainty_threshold,
            "review_queue": review_queue[:50],
            "all_scores": rows[:100],
            "summary": f"{len(review_queue)}/{len(rows)} rows need manual review (confidence < {self.uncertainty_threshold})",
        }
