"""Tests for ML Intelligence services."""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.dataset_quality import DatasetQualityScorer
from app.services.experiment_comparator import ExperimentComparator
from app.services.feature_selector import FeatureSelector
from app.services.leakage_detector import LeakageDetector


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0] * 30,
        "sepal_width": [3.5, 3.0, 3.2, 3.1, 3.6] * 30,
        "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4] * 30,
        "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2] * 30,
        "species": ["setosa", "setosa", "setosa", "setosa", "setosa"] * 30,
    })


def test_leakage_detects_id_column(sample_df):
    df = sample_df.copy()
    df["customer_id"] = range(len(df))
    report = LeakageDetector().detect(df, "species")
    assert report["leakage_detected"]
    assert "customer_id" in report["recommended_drop"]


def test_leakage_clean_dataset(sample_df):
    report = LeakageDetector().detect(sample_df, "species")
    assert not report["leakage_detected"] or report["n_issues"] == 0


def test_dataset_quality_score(sample_df):
    result = DatasetQualityScorer().score(sample_df, "species")
    assert 0 <= result["overall_score"] <= 100
    assert "dimensions" in result
    assert result["suggestions"]


def test_feature_selection(sample_df):
    X = sample_df.drop(columns=["species"])
    y = sample_df["species"]
    X_sel, report = FeatureSelector(max_features=3).select(X, y, "classification")
    assert report["selected_count"] <= 3
    assert len(X_sel.columns) == report["selected_count"]


def test_experiment_compare(monkeypatch):
    fake_registry = {
        "job_a": {
            "model_name": "random_forest",
            "dataset_id": "ds_1",
            "target_column": "y",
            "metrics": {"accuracy": 0.9, "roc_auc": 0.85},
            "best_params": {"n_estimators": 100},
            "baseline_scores": {"random_forest": 0.88},
            "top_features": [{"feature": "x1", "importance": 0.5}],
        },
        "job_b": {
            "model_name": "catboost",
            "dataset_id": "ds_1",
            "target_column": "y",
            "metrics": {"accuracy": 0.93, "roc_auc": 0.91},
            "best_params": {"depth": 6},
            "baseline_scores": {"catboost": 0.92},
            "top_features": [{"feature": "x1", "importance": 0.4}],
        },
    }

    class FakeRegistry:
        def get(self, job_id):
            return fake_registry.get(job_id)

    monkeypatch.setattr("app.services.experiment_comparator.get_model_registry", lambda: FakeRegistry())
    result = ExperimentComparator().compare("job_a", "job_b")
    assert result["run_a"]["job_id"] == "job_a"
    assert any(d["metric"] == "accuracy" for d in result["metric_diffs"])
