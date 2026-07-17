"""Smoke tests for the FastAPI surface (upload + profile endpoints)."""
import io

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "AutoMLOps Platform"


def test_upload_and_profile_dataset(classification_df):
    csv_bytes = classification_df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/datasets/upload",
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n_rows"] == len(classification_df)
    dataset_id = body["dataset_id"]

    profile_response = client.get(f"/datasets/{dataset_id}/profile")
    assert profile_response.status_code == 200
    profile = profile_response.json()["profile"]
    assert profile["n_rows"] == len(classification_df)


def test_upload_rejects_non_csv():
    response = client.post(
        "/datasets/upload",
        files={"file": ("data.txt", io.BytesIO(b"not a csv"), "text/plain")},
    )
    assert response.status_code == 400


def test_predict_returns_404_for_unknown_job():
    response = client.post("/predict/unknown_job_id", json={"records": [{"a": 1}]})
    assert response.status_code == 404


def test_root_includes_ui_link():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body.get("ui") == "/ui/"


def test_ui_served():
    response = client.get("/ui/")
    assert response.status_code == 200
    assert "AutoMLOps Platform" in response.text


def test_ui_redirect():
    response = client.get("/ui", follow_redirects=False)
    assert response.status_code in (307, 308)
    assert response.headers.get("location") == "/ui/"


def test_agent_list_models():
    response = client.get("/agent/models")
    assert response.status_code == 200
    body = response.json()
    assert "classification" in body
    assert "gradient_boosting" in body["classification"]


def test_agent_knowledge_topics():
    response = client.get("/agent/knowledge")
    assert response.status_code == 200
    topics = response.json()
    assert len(topics) >= 5


def test_agent_analyze_with_profile():
    response = client.post(
        "/agent/analyze",
        json={
            "profile": {
                "n_rows": 100,
                "n_columns": 5,
                "numeric_columns": ["a", "b"],
                "categorical_columns": ["c"],
                "datetime_columns": [],
                "missing_values": {},
                "n_duplicate_rows": 0,
                "target_analysis": {"type": "classification"},
            },
            "target_column": "churn",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "pre_train"
    assert body["task_type"] == "classification"
    assert len(body["model_recommendations"]) >= 1
    assert "narrative_report" in body
    assert "confidence" in body


def test_chat_status():
    response = client.get("/chat/status")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert "provider" in body
    assert body["provider"] in ("openai", "gemini", "groq", "ollama", "rules")


def test_rules_chat_help(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "rules")
    response = client.post("/chat/message", json={"message": "help"})
    assert response.status_code == 200
    body = response.json()
    assert "list jobs" in body["response"].lower()
    assert body.get("provider") == "rules"


def test_eda_endpoint(classification_df):
    import io
    from app.core.config import get_settings

    csv_bytes = classification_df.to_csv(index=False).encode("utf-8")
    upload = client.post(
        "/datasets/upload",
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]
    response = client.get(f"/datasets/{dataset_id}/eda?target_column=churned")
    assert response.status_code == 200
    body = response.json()
    assert body["target_column"] == "churned"
    assert "target_analysis" in body["eda"]
