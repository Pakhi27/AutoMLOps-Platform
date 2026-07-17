"""Compare two training runs side-by-side."""
from __future__ import annotations

from typing import Any

from app.services.model_registry import get_model_registry


class ExperimentComparator:
    def compare(self, job_id_a: str, job_id_b: str) -> dict[str, Any]:
        registry = get_model_registry()
        a = registry.get(job_id_a)
        b = registry.get(job_id_b)
        if a is None:
            raise ValueError(f"Job '{job_id_a}' not found")
        if b is None:
            raise ValueError(f"Job '{job_id_b}' not found")

        metrics_a = a.get("metrics") or {}
        metrics_b = b.get("metrics") or {}
        all_metric_keys = sorted(set(metrics_a) | set(metrics_b))

        metric_diffs = []
        for key in all_metric_keys:
            va = metrics_a.get(key)
            vb = metrics_b.get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                delta = vb - va
                pct = (delta / abs(va) * 100) if va else None
                metric_diffs.append({
                    "metric": key,
                    "run_a": round(float(va), 4),
                    "run_b": round(float(vb), 4),
                    "delta": round(delta, 4),
                    "pct_change": round(pct, 2) if pct is not None else None,
                    "winner": job_id_b if delta > 0 else job_id_a if delta < 0 else "tie",
                })

        params_a = a.get("best_params") or {}
        params_b = b.get("best_params") or {}
        param_changes = []
        for key in sorted(set(params_a) | set(params_b)):
            if params_a.get(key) != params_b.get(key):
                param_changes.append({
                    "param": key,
                    "run_a": params_a.get(key),
                    "run_b": params_b.get(key),
                })

        fi_a = {f["feature"]: f.get("importance", 0) for f in (a.get("top_features") or [])}
        fi_b = {f["feature"]: f.get("importance", 0) for f in (b.get("top_features") or [])}
        fi_changes = []
        for feat in sorted(set(fi_a) | set(fi_b))[:10]:
            if fi_a.get(feat) != fi_b.get(feat):
                fi_changes.append({
                    "feature": feat,
                    "importance_a": fi_a.get(feat),
                    "importance_b": fi_b.get(feat),
                })

        return {
            "run_a": self._run_summary(job_id_a, a),
            "run_b": self._run_summary(job_id_b, b),
            "same_dataset": a.get("dataset_id") == b.get("dataset_id"),
            "same_target": a.get("target_column") == b.get("target_column"),
            "model_changed": a.get("model_name") != b.get("model_name"),
            "metric_diffs": metric_diffs,
            "hyperparameter_changes": param_changes,
            "feature_importance_changes": fi_changes,
            "summary": self._summary(job_id_a, job_id_b, a, b, metric_diffs),
        }

    @staticmethod
    def _run_summary(job_id: str, entry: dict) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "model_name": entry.get("model_name"),
            "dataset_id": entry.get("dataset_id"),
            "target_column": entry.get("target_column"),
            "metrics": entry.get("metrics"),
            "elapsed_seconds": entry.get("elapsed_seconds"),
        }

    @staticmethod
    def _summary(job_a: str, job_b: str, a: dict, b: dict, diffs: list[dict]) -> str:
        parts = []
        if a.get("model_name") != b.get("model_name"):
            parts.append(f"Model: {a.get('model_name')} → {b.get('model_name')}")
        for d in diffs[:3]:
            if d.get("pct_change") is not None:
                parts.append(f"{d['metric']}: {d['pct_change']:+.1f}%")
        return " · ".join(parts) if parts else "Runs are structurally similar — compare metrics below."
