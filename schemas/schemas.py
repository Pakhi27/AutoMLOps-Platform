"""Pydantic request/response models shared across the API layer."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"


class DataModality(str, Enum):
    tabular = "tabular"
    text = "text"
    image = "image"
    timeseries = "timeseries"
    logs = "logs"
    documents = "documents"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    success = "success"
    failed = "failed"


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    n_rows: int
    n_columns: int
    columns: list[str]
    modality: str = "tabular"
    pipeline_type: str = "tabular_automl"
    detection_reason: Optional[str] = None
    suggested_targets: list[str] = Field(default_factory=list)
    text_column: Optional[str] = None
    datetime_column: Optional[str] = None


class ModalityInfoResponse(BaseModel):
    dataset_id: str
    modality: str
    pipeline_type: str
    detection_reason: Optional[str] = None
    preprocessing: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    explainability: list[str] = Field(default_factory=list)
    drift: list[str] = Field(default_factory=list)
    suggested_targets: list[str] = Field(default_factory=list)
    text_column: Optional[str] = None
    datetime_column: Optional[str] = None


class DatabaseConnectRequest(BaseModel):
    connection_url: str = Field(..., description="SQLAlchemy URL, e.g. sqlite:///db.sqlite or postgresql://user:pass@host/db")
    table: str
    schema: Optional[str] = None
    query: Optional[str] = Field(default=None, description="Custom SELECT instead of table")
    limit: int = Field(default=100_000, ge=100, le=1_000_000)


class PipelineRunRequest(BaseModel):
    dataset_id: str
    target_column: str
    modality: Optional[DataModality] = Field(default=None, description="Override auto-detected modality.")
    text_column: Optional[str] = Field(default=None, description="For text/logs/documents pipelines.")
    datetime_column: Optional[str] = Field(default=None, description="For time-series pipelines.")
    task_type: Optional[TaskType] = Field(
        default=None, description="Auto-detected from the target column if omitted."
    )
    n_trials: Optional[int] = Field(default=None, description="Optuna trials; falls back to config default.")
    test_size: Optional[float] = Field(default=None, ge=0.05, le=0.5)


class PipelineRunAccepted(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    modality: Optional[str] = None
    pipeline_type: Optional[str] = None


class ProfileResponse(BaseModel):
    dataset_id: str
    profile: dict[str, Any]


class JobRecord(BaseModel):
    job_id: str
    dataset_id: str
    target_column: str
    status: JobStatus
    task_type: Optional[str] = None
    modality: Optional[str] = None
    pipeline_type: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    progress: Optional[dict[str, Any]] = None
    progress_log: Optional[list[dict[str, Any]]] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="List of raw feature rows (JSON records).")
    explain: bool = Field(default=False, description="Include per-row SHAP contributions.")


class PredictResponse(BaseModel):
    job_id: str
    model_name: str
    task_type: str
    predictions: list[Any]
    probabilities: Optional[list[Any]] = None
    explanations: Optional[list[dict[str, Any]]] = None
    modality: Optional[str] = None
    label_classes: Optional[list[str]] = None
    image_rows: Optional[list[dict[str, Any]]] = None


class DriftSummary(BaseModel):
    job_id: str
    dataset_drift_detected: bool
    drift_share: float
    number_of_drifted_columns: int
    number_of_columns: int
    report_html_path: str
    drifted_columns: list[str]
    retrain_recommended: bool = False
    retrain_job_id: Optional[str] = None
    retrain_message: Optional[str] = None


class EDAResponse(BaseModel):
    dataset_id: str
    target_column: str
    eda: dict[str, Any]


class FeatureImportanceResponse(BaseModel):
    job_id: str
    features: list[dict[str, Any]]


class BatchPredictResponse(BaseModel):
    job_id: str
    model_name: str
    task_type: str
    n_rows: int
    download_path: str
    preview: list[dict[str, Any]]


class KnowledgeTopic(BaseModel):
    source: str
    title: str


class AgentAnalyzeRequest(BaseModel):
    dataset_id: Optional[str] = Field(default=None, description="Uploaded dataset ID to analyze.")
    job_id: Optional[str] = Field(default=None, description="Post-train mode: analyze completed job with evidence.")
    profile: Optional[dict[str, Any]] = Field(default=None, description="Raw profile dict (alternative to dataset_id).")
    target_column: Optional[str] = Field(default=None)
    task_type: Optional[TaskType] = Field(default=None)
    query: Optional[str] = Field(default=None, description="Custom RAG search query.")


class AgentAnalyzeResponse(BaseModel):
    mode: Optional[str] = None
    task_type: Optional[str] = None
    confidence: float = 0.5
    critic_passed: bool = True
    fingerprint: Optional[dict[str, Any]] = None
    data_insights: list[str] = Field(default_factory=list)
    model_recommendations: list[str] = Field(default_factory=list)
    model_recommendations_detail: list[dict[str, Any]] = Field(default_factory=list)
    preprocessing_tips: list[str] = Field(default_factory=list)
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)
    top_actions: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    narrative_report: str = ""
    llm_used: bool = False
    llm_provider: Optional[str] = None
    web_evidence_used: bool = False


class LeakageReportResponse(BaseModel):
    dataset_id: str
    target_column: str
    report: dict[str, Any]


class DatasetQualityResponse(BaseModel):
    dataset_id: str
    target_column: str
    quality: dict[str, Any]


class ModelReviewResponse(BaseModel):
    job_id: str
    review: dict[str, Any]


class BusinessInsightsResponse(BaseModel):
    job_id: str
    insights: dict[str, Any]


class ModelCardResponse(BaseModel):
    job_id: str
    card: dict[str, Any]
    markdown: str


class ExperimentCompareResponse(BaseModel):
    comparison: dict[str, Any]


class CounterfactualRequest(BaseModel):
    record: dict[str, Any]


class CounterfactualResponse(BaseModel):
    job_id: str
    result: dict[str, Any]


class ActiveLearningResponse(BaseModel):
    job_id: str
    result: dict[str, Any]


class FeatureDriftResponse(BaseModel):
    job_id: str
    feature_drift: list[dict[str, Any]]
    dataset_drift: dict[str, Any]
