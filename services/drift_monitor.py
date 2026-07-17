"""Stage 9: Data drift detection with Evidently AI.

Uses Evidently's modern (0.7+) Report API: `evidently.Report` +
`evidently.presets.DataDriftPreset`. A reference dataset (the cleaned
training slice captured at pipeline-run time) is compared against a new
"current" dataset supplied later via the /monitor/drift endpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


class DriftMonitor:
    def __init__(self, dataset_drift_share_threshold: float = 0.5):
        self.dataset_drift_share_threshold = dataset_drift_share_threshold

    def run_drift_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        output_html_path: str | Path,
    ) -> dict[str, Any]:
        common_cols = [c for c in reference.columns if c in current.columns]
        reference = reference[common_cols]
        current = current[common_cols]

        report = Report([DataDriftPreset()])
        snapshot = report.run(current_data=current, reference_data=reference)

        output_html_path = Path(output_html_path)
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_html_path))

        try:
            result_dict = snapshot.dict()
            summary = self._parse_summary(result_dict, len(common_cols))
        except Exception:
            summary = {
                "dataset_drift_detected": False,
                "drift_share": 0.0,
                "number_of_drifted_columns": 0,
                "number_of_columns": len(common_cols),
                "drifted_columns": [],
            }
        summary["report_html_path"] = str(output_html_path)
        summary["feature_drift"] = self._feature_drift_details(reference, current)
        return summary

    def get_feature_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        return self._feature_drift_details(reference, current)

    def _feature_drift_details(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Per-feature drift using PSI-like bin comparison for numerics, category shift for categoricals."""
        common = [c for c in reference.columns if c in current.columns]
        results: list[dict[str, Any]] = []
        for col in common:
            ref = reference[col].dropna()
            cur = current[col].dropna()
            if ref.empty or cur.empty:
                results.append({"feature": col, "drift_level": "unknown", "score": 0.0, "detail": "Insufficient data"})
                continue
            if pd.api.types.is_numeric_dtype(ref):
                score = self._numeric_drift_score(ref, cur)
            else:
                score = self._categorical_drift_score(ref.astype(str), cur.astype(str))
            level = "high" if score >= 0.25 else "low" if score >= 0.1 else "none"
            results.append({
                "feature": col,
                "drift_level": level,
                "score": round(score, 4),
                "detail": f"Distribution shift score {score:.3f}",
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    @staticmethod
    def _numeric_drift_score(ref: pd.Series, cur: pd.Series) -> float:
        try:
            bins = np.unique(np.percentile(ref, [0, 25, 50, 75, 100]))
            if len(bins) < 2:
                return 0.0
            ref_hist, _ = np.histogram(ref, bins=bins)
            cur_hist, _ = np.histogram(cur, bins=bins)
            ref_pct = ref_hist / max(ref_hist.sum(), 1)
            cur_pct = cur_hist / max(cur_hist.sum(), 1)
            return float(np.sum(np.abs(ref_pct - cur_pct)) / 2)
        except Exception:
            return abs(ref.mean() - cur.mean()) / (ref.std() + 1e-6) * 0.1

    @staticmethod
    def _categorical_drift_score(ref: pd.Series, cur: pd.Series) -> float:
        ref_dist = ref.value_counts(normalize=True)
        cur_dist = cur.value_counts(normalize=True)
        all_cats = set(ref_dist.index) | set(cur_dist.index)
        return float(sum(abs(ref_dist.get(c, 0) - cur_dist.get(c, 0)) for c in all_cats) / 2)

    def _parse_summary(self, result_dict: dict[str, Any], n_columns: int) -> dict[str, Any]:
        """Best-effort extraction of the headline drift numbers.

        Evidently's result payload nests metric outputs inside `metrics`, a
        list of {metric_id / metric, value, ...} entries. We scan for the
        dataset-level drift share and per-column drift flags defensively so
        this keeps working across minor point-release schema tweaks.
        """
        drift_share = 0.0
        n_drifted = 0
        drifted_columns: list[str] = []

        metrics = result_dict.get("metrics", [])
        for m in metrics:
            metric_id = str(m.get("metric_id") or m.get("metric") or "")
            value = m.get("value")
            if "DriftedColumnsCount" in metric_id or "DatasetDriftMetric" in metric_id:
                if isinstance(value, dict):
                    drift_share = float(value.get("share", drift_share))
                    n_drifted = int(value.get("count", n_drifted))
            if "ValueDrift" in metric_id and isinstance(value, (int, float)) and value:
                # column-level drift metric ids typically embed the column name
                col_name = m.get("column") or m.get("parameters", {}).get("column")
                if col_name and value >= 0.5:
                    drifted_columns.append(str(col_name))

        if not n_columns:
            n_columns = 1
        dataset_drift_detected = drift_share >= self.dataset_drift_share_threshold or bool(drifted_columns)

        return {
            "dataset_drift_detected": bool(dataset_drift_detected),
            "drift_share": round(float(drift_share), 4),
            "number_of_drifted_columns": n_drifted or len(drifted_columns),
            "number_of_columns": n_columns,
            "drifted_columns": drifted_columns,
        }
