"""Post-train advisor excludes structurally unrelated past runs."""
from app.services.ml_agent_graph import MLAdvisorAgent


def test_post_train_advisor_structural_filtering():
    profile = {
        "n_rows": 150,
        "n_columns": 5,
        "numeric_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "categorical_columns": ["species"],
        "datetime_columns": [],
        "missing_values": {},
        "target_analysis": {"type": "classification", "is_imbalanced": False},
        "correlation_with_target": [{"feature": "petal_width", "correlation": 0.9565}],
    }
    job_result = {
        "model_name": "random_forest",
        "task_type": "classification",
        "metrics": {"roc_auc": 0.9867, "f1_weighted": 0.9333},
        "baseline_scores": {"random_forest": 0.9569, "catboost": 0.9569},
        "best_params": {"n_estimators": 250},
        "feature_importance": [
            {"feature": "petal_width", "importance": 0.45},
            {"feature": "petal_length", "importance": 0.35},
        ],
    }
    result = MLAdvisorAgent().analyze(
        profile=profile,
        target_column="species",
        job_result=job_result,
        job_id="job_iris_test",
    )

    action_text = " ".join(a["action"] for a in result["top_actions"]).lower()
    assert "churn" not in action_text

    for doc in result["retrieved_docs"]:
        if doc.get("chunk_type") != "run_memory":
            continue
        assert (doc.get("target_column") or "").lower() in ("", "species")

    tips = " ".join(result["preprocessing_tips"]).lower()
    assert "charges" not in tips

    assert result["fingerprint"]["target_column"] == "species"
    assert "dataset_signature" in result["fingerprint"]
