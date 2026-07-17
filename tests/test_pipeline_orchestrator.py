"""End-to-end test of the full AutoML pipeline (kept fast via tiny CV/trial counts)."""
import shutil
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.services.model_registry import get_model_registry
from app.services.pipeline_orchestrator import PipelineOrchestrator
from app.utils.io_utils import new_id


@pytest.fixture
def fast_settings():
    settings = get_settings()
    original_cv = settings.modeling.get("cv_folds")
    original_clf = settings.modeling.get("classification_candidates")
    original_reg = settings.modeling.get("regression_candidates")
    settings.modeling["cv_folds"] = 2
    settings.modeling["classification_candidates"] = [
        "logistic_regression", "random_forest", "xgboost",
    ]
    settings.modeling["regression_candidates"] = ["ridge", "random_forest", "xgboost"]
    yield settings
    settings.modeling["cv_folds"] = original_cv
    settings.modeling["classification_candidates"] = original_clf
    settings.modeling["regression_candidates"] = original_reg


def _write_dataset(settings, df) -> str:
    dataset_id = new_id("ds_test_")
    df.to_csv(settings.upload_dir / f"{dataset_id}.csv", index=False)
    return dataset_id


def _cleanup(settings, dataset_id: str, job_id: str) -> None:
    (settings.upload_dir / f"{dataset_id}.csv").unlink(missing_ok=True)
    (settings.reference_dir / f"{job_id}.csv").unlink(missing_ok=True)
    shutil.rmtree(settings.artifacts_dir / job_id, ignore_errors=True)


def test_full_pipeline_classification(fast_settings, classification_df):
    dataset_id = _write_dataset(fast_settings, classification_df)
    job_id = new_id("job_test_")
    orchestrator = PipelineOrchestrator()
    try:
        result = orchestrator.run(job_id=job_id, dataset_id=dataset_id, target_column="churned", n_trials=2)
        assert result["task_type"] == "classification"
        assert "accuracy" in result["metrics"]
        assert get_model_registry().get(job_id) is not None
        assert result["shap_plot_path"] is not None
        assert Path(result["shap_plot_path"]).exists()
    finally:
        _cleanup(fast_settings, dataset_id, job_id)


def test_full_pipeline_regression(fast_settings, regression_df):
    dataset_id = _write_dataset(fast_settings, regression_df)
    job_id = new_id("job_test_")
    orchestrator = PipelineOrchestrator()
    try:
        result = orchestrator.run(job_id=job_id, dataset_id=dataset_id, target_column="target", n_trials=2)
        assert result["task_type"] == "regression"
        assert "r2" in result["metrics"]
        assert result["shap_plot_path"] is not None
        assert Path(result["shap_plot_path"]).exists()
    finally:
        _cleanup(fast_settings, dataset_id, job_id)
