"""Tests for EDA, feature importance, and evidence-grounded advisor."""
import pandas as pd

from app.services.eda_service import EDAService
from app.services.ml_agent_graph import MLAdvisorAgent


def test_eda_classification(classification_df):
    eda = EDAService().analyze(classification_df, "churned")
    assert eda["task_type"] == "classification"
    assert "target_analysis" in eda
    assert eda["target_analysis"]["type"] == "classification"
    assert len(eda["correlation_with_target"]) >= 0
    assert "age" in eda["numeric_features"] or len(eda["numeric_features"]) > 0


def test_eda_regression(regression_df):
    eda = EDAService().analyze(regression_df, "target")
    assert eda["task_type"] == "regression"
    assert eda["target_analysis"]["type"] == "regression"


def test_post_train_advisor():
    profile = {
        "n_rows": 500,
        "n_columns": 10,
        "numeric_columns": ["age", "tenure_months"],
        "categorical_columns": ["contract"],
        "datetime_columns": [],
        "missing_values": {},
        "target_analysis": {"type": "classification", "is_imbalanced": True, "imbalance_ratio": 0.72},
        "correlation_with_target": [{"feature": "tenure_months", "correlation": -0.35}],
    }
    job_result = {
        "model_name": "gradient_boosting",
        "task_type": "classification",
        "metrics": {"accuracy": 0.71, "roc_auc": 0.79},
        "baseline_scores": {"gradient_boosting": 0.70, "xgboost": 0.69, "svm": 0.68},
    }
    agent = MLAdvisorAgent()
    result = agent.analyze(profile=profile, target_column="churn", job_result=job_result)
    assert result["mode"] == "post_train"
    assert result["confidence"] > 0
    assert len(result["data_insights"]) >= 2
    assert "gradient_boosting" in result["model_recommendations"]
    assert len(result["narrative_report"]) > 80
