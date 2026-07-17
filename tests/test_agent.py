"""Tests for RAG knowledge base and LangGraph ML advisor."""
from app.services.ml_agent_graph import MLAdvisorAgent
from app.services.model_selector import ModelSelector
from app.services.rag_knowledge import get_knowledge_base


def test_available_models_includes_new_algorithms():
    clf = ModelSelector.available_models("classification")
    reg = ModelSelector.available_models("regression")
    assert "gradient_boosting" in clf
    assert "hist_gradient_boosting" in clf
    assert "svm" in clf
    assert "knn" in clf
    assert "elastic_net" in reg


def test_rag_knowledge_retrieval():
    kb = get_knowledge_base()
    topics = kb.list_topics()
    assert len(topics) >= 7
    chunks = kb.retrieve("churn classification model selection imbalance", top_k=3)
    assert len(chunks) > 0
    merged = kb.retrieve_merged(
        {"models": "churn classification", "imbalance": "class imbalance f1"},
        top_k_per_query=2,
        context={
            "task_type": "classification",
            "target_column": "churn",
            "column_tokens": ["churn", "tenure", "monthly", "charges", "contract"],
            "is_imbalanced": True,
        },
    )
    assert len(merged) >= 2


def test_ml_advisor_agent_runs():
    profile = {
        "n_rows": 500,
        "n_columns": 10,
        "numeric_columns": ["age", "tenure_months", "monthly_charges"],
        "categorical_columns": ["contract", "internet_service"],
        "datetime_columns": ["signup_date"],
        "missing_values": {},
        "n_duplicate_rows": 0,
        "target_analysis": {"type": "classification"},
    }
    agent = MLAdvisorAgent()
    result = agent.analyze(profile=profile, target_column="churn")
    assert result["mode"] == "pre_train"
    assert result["task_type"] == "classification"
    assert len(result["model_recommendations"]) >= 1
    assert len(result["data_insights"]) >= 1
    assert len(result["retrieved_docs"]) >= 3
    assert len(result["preprocessing_tips"]) >= 1
    assert result["fingerprint"]["target_column"] == "churn"
    assert len(result["narrative_report"]) > 80
    assert "churn" in result["narrative_report"].lower() or "classification" in result["narrative_report"].lower()
