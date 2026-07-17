"""Stage 7: Experiment tracking with MLflow.

Thin wrapper around the MLflow client so the orchestrator doesn't need to
know about tracking URIs / experiment bookkeeping. Logs params, metrics,
artifacts (profiling/cleaning/drift reports, SHAP plots) and the final
sklearn Pipeline (preprocessing + model) as an MLflow model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

from app.core.config import get_settings


class ExperimentTracker:
    def __init__(self) -> None:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    def start_run(self, run_name: str):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        safe = {k: v for k, v in params.items() if v is not None}
        mlflow.log_params(safe)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})

    def log_dict_artifact(self, data: dict[str, Any], filename: str) -> None:
        mlflow.log_dict(data, filename)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_model(self, pipeline, artifact_path: str = "model", registered_model_name: str | None = None) -> str:
        info = mlflow.sklearn.log_model(
            pipeline,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
        return info.model_uri

    def set_tags(self, tags: dict[str, Any]) -> None:
        mlflow.set_tags(tags)

    def active_run_id(self) -> str:
        run = mlflow.active_run()
        return run.info.run_id if run else ""

    def end_run(self, status: str = "FINISHED") -> None:
        mlflow.end_run(status=status)
