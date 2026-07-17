"""Tests for config-driven advisor relevance (no hardcoded business domains)."""
from app.services.advisor_relevance import (
    comparable_run_actions,
    filter_run_memory_docs,
    filter_tips_for_dataset,
    score_run_memory_relevance,
)
from app.services.dataset_fingerprint import build_dataset_context, build_dataset_signature


def _churn_run(job_id: str = "job_churn_1") -> dict:
    return {
        "source": f"run:{job_id}",
        "title": "Past run — gradient_boosting on churn",
        "chunk_type": "run_memory",
        "task_type": "classification",
        "target_column": "churn",
        "feature_columns": ["tenure_months", "monthly_charges", "contract"],
        "top_features": ["tenure_months", "monthly_charges"],
        "dataset_signature": build_dataset_signature(
            ["tenure_months", "monthly_charges", "contract", "churn"], "churn"
        ),
        "n_rows": 500,
        "row_bucket": "small",
    }


def _species_run(job_id: str = "job_iris_1") -> dict:
    return {
        "source": f"run:{job_id}",
        "title": "Past run — random_forest on species",
        "chunk_type": "run_memory",
        "task_type": "classification",
        "target_column": "species",
        "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "top_features": ["petal_width", "petal_length"],
        "dataset_signature": build_dataset_signature(
            ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"], "species"
        ),
        "n_rows": 150,
        "row_bucket": "small",
    }


def _iris_context() -> dict:
    profile = {
        "n_rows": 150,
        "n_columns": 5,
        "numeric_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "categorical_columns": ["species"],
        "datetime_columns": [],
        "missing_values": {},
        "target_analysis": {"type": "classification"},
    }
    return build_dataset_context(profile, "species").to_rag_context()


def test_structural_similarity_prefers_matching_target_and_features():
    ctx = _iris_context()
    churn_score = score_run_memory_relevance(_churn_run(), ctx)
    species_score = score_run_memory_relevance(_species_run(), ctx)
    assert species_score > churn_score
    assert churn_score < 0.48


def test_filter_run_memory_keeps_structurally_similar_runs_only():
    ctx = _iris_context()
    docs = [_churn_run(), _species_run(), {"chunk_type": "playbook", "title": "Tuning", "content": "optuna"}]
    filtered = filter_run_memory_docs(docs, ctx)
    run_titles = [d["title"] for d in filtered if d.get("chunk_type") == "run_memory"]
    assert any("species" in t for t in run_titles)
    assert not any("churn" in t for t in run_titles)


def test_comparable_actions_capped_and_deduplicated():
    ctx = _iris_context()
    docs = filter_run_memory_docs([_churn_run(), _species_run()], ctx)
    actions = comparable_run_actions(docs, max_actions=2)
    assert len(actions) <= 2
    assert all("churn" not in a["action"].lower() for a in actions)


def test_tip_filter_uses_vocabulary_not_domain_labels():
    tokens = build_dataset_context(
        {
            "n_rows": 150,
            "n_columns": 5,
            "numeric_columns": ["petal_width", "petal_length"],
            "categorical_columns": ["species"],
            "datetime_columns": [],
            "missing_values": {},
        },
        "species",
    ).column_tokens
    tips = [
        "Run cross-validated baseline on all candidates.",
        "Numeric: check skew — log1p for right-skewed charges, amounts, counts",
        "Compare petal_width distribution across species classes.",
    ]
    filtered = filter_tips_for_dataset(tips, tokens)
    assert any("baseline" in t for t in filtered)
    assert not any("charges" in t for t in filtered)
    assert any("petal" in t for t in filtered)
