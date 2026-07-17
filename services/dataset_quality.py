"""Composite dataset quality score (0–100) with improvement suggestions."""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.data_profiler import DataProfiler
from app.services.eda_service import EDAService
from app.services.leakage_detector import LeakageDetector


class DatasetQualityScorer:
    def score(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        profile = DataProfiler().profile(df)
        eda = EDAService().analyze(df, target_column)
        leakage = LeakageDetector().detect(df, target_column)

        dimensions: dict[str, dict[str, Any]] = {}

        # Missing values
        missing_pcts = [v["pct_missing"] for v in profile.get("missing_values", {}).values()]
        avg_missing = sum(missing_pcts) / len(missing_pcts) if missing_pcts else 0
        miss_score = max(0, 100 - avg_missing * 200)
        dimensions["missing_values"] = {
            "score": round(miss_score, 1),
            "grade": _grade(miss_score),
            "detail": f"Avg missing {avg_missing * 100:.1f}%",
        }

        # Duplicates
        dup_pct = profile.get("n_duplicate_rows", 0) / max(len(df), 1)
        dup_score = max(0, 100 - dup_pct * 500)
        dimensions["duplicates"] = {
            "score": round(dup_score, 1),
            "grade": _grade(dup_score),
            "detail": f"{profile.get('n_duplicate_rows', 0)} duplicate rows",
        }

        # Outliers (numeric cols with extreme skew)
        numeric_summary = profile.get("numeric_summary", {})
        skews = [abs(v.get("skew", 0)) for v in numeric_summary.values()]
        avg_skew = sum(skews) / len(skews) if skews else 0
        outlier_score = max(0, 100 - min(avg_skew, 5) * 12)
        dimensions["outliers"] = {
            "score": round(outlier_score, 1),
            "grade": _grade(outlier_score),
            "detail": f"Avg |skew| {avg_skew:.2f}",
        }

        # Leakage
        leak_score = 100 if not leakage["leakage_detected"] else max(20, 100 - leakage["n_issues"] * 15)
        dimensions["leakage"] = {
            "score": round(leak_score, 1),
            "grade": _grade(leak_score),
            "detail": leakage["summary"],
        }

        # Target balance
        ta = eda.get("target_analysis", {})
        if ta.get("is_imbalanced"):
            ratio = ta.get("imbalance_ratio", 0.5)
            balance_score = max(30, min(100, ratio * 200))
            balance_detail = f"Majority class {ratio * 100:.0f}%"
        else:
            balance_score = 95
            balance_detail = "Well balanced" if ta.get("type") == "classification" else "Regression target"
        dimensions["target_balance"] = {
            "score": round(balance_score, 1),
            "grade": _grade(balance_score),
            "detail": balance_detail,
        }

        # Sample size
        n_rows = len(df)
        size_score = min(100, 40 + n_rows / 50)
        dimensions["sample_size"] = {
            "score": round(size_score, 1),
            "grade": _grade(size_score),
            "detail": f"{n_rows:,} rows",
        }

        weights = {
            "missing_values": 0.2,
            "duplicates": 0.1,
            "outliers": 0.15,
            "leakage": 0.25,
            "target_balance": 0.15,
            "sample_size": 0.15,
        }
        overall = sum(dimensions[k]["score"] * weights[k] for k in weights)
        suggestions = _suggestions(dimensions, leakage, profile, target_column)

        return {
            "overall_score": round(overall, 1),
            "grade": _grade(overall),
            "dimensions": dimensions,
            "leakage": leakage,
            "suggestions": suggestions,
            "n_rows": n_rows,
            "n_columns": profile.get("n_columns", len(df.columns)),
        }


def _grade(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Moderate"
    if score >= 40:
        return "Fair"
    return "Poor"


def _suggestions(
    dimensions: dict,
    leakage: dict,
    profile: dict,
    target_column: str,
) -> list[str]:
    tips: list[str] = []
    if dimensions["missing_values"]["score"] < 80:
        tips.append("Impute or drop columns with high missingness before training.")
    if dimensions["duplicates"]["score"] < 90:
        tips.append("Remove duplicate rows — they inflate metrics artificially.")
    if dimensions["outliers"]["score"] < 70:
        tips.append("Apply outlier capping or robust scaling on skewed numeric features.")
    if leakage.get("recommended_drop"):
        cols = ", ".join(leakage["recommended_drop"][:5])
        tips.append(f"Remove leakage columns before training: {cols}")
    if dimensions["target_balance"]["score"] < 70:
        tips.append("Use stratified CV and class-weighted models for imbalanced target.")
    if dimensions["sample_size"]["score"] < 60:
        tips.append("Collect more data or use simpler models with strong regularization.")
    if profile.get("constant_columns"):
        tips.append(f"Drop constant columns: {', '.join(profile['constant_columns'][:5])}")
    if not tips:
        tips.append("Dataset quality looks good — proceed to training.")
    return tips
