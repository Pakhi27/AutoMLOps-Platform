"""Data drift monitoring with optional auto-retrain."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.schemas import DriftSummary, JobStatus
from app.services.drift_monitor import DriftMonitor
from app.services.model_registry import get_model_registry
from app.store.job_store import get_job_store
from app.utils.io_utils import new_id, read_csv_safely

router = APIRouter(prefix="/monitor", tags=["monitor"])
logger = get_logger(__name__)


def _trigger_retrain(
    parent_job_id: str,
    dataset_id: str,
    target_column: str,
    new_job_id: str,
    n_trials: int,
) -> None:
    from app.services.pipeline_orchestrator import PipelineOrchestrator

    store = get_job_store()
    store.create(new_job_id, dataset_id, target_column)
    store.update(new_job_id, status=JobStatus.running.value, parent_job_id=parent_job_id)
    try:
        result = PipelineOrchestrator().run(
            job_id=new_job_id,
            dataset_id=dataset_id,
            target_column=target_column,
            n_trials=n_trials,
        )
        store.update(new_job_id, status=JobStatus.success.value, result=result, task_type=result.get("task_type"))
        logger.info("Auto-retrain %s completed (triggered by drift on %s)", new_job_id, parent_job_id)
    except Exception as exc:
        logger.exception("Auto-retrain %s failed", new_job_id)
        store.update(new_job_id, status=JobStatus.failed.value, error=str(exc))


@router.post("/drift/{job_id}", response_model=DriftSummary)
async def check_drift(
    job_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_retrain: bool = Query(False, description="Automatically start retraining if drift exceeds threshold."),
    n_trials: int = Query(15, ge=2, le=100),
) -> DriftSummary:
    settings = get_settings()
    registry = get_model_registry()
    entry = registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No trained model / reference data found for job '{job_id}'.")

    reference_path = settings.reference_dir / f"{job_id}.csv"
    if not reference_path.exists():
        raise HTTPException(status_code=404, detail="Reference dataset missing for this job.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    tmp_path = settings.artifacts_dir / job_id / "current_upload.csv"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(contents)

    reference_df = read_csv_safely(reference_path)
    current_df = read_csv_safely(tmp_path)

    threshold = settings.drift.get("dataset_drift_share_threshold", 0.5)
    monitor = DriftMonitor(dataset_drift_share_threshold=threshold)
    output_html_path = settings.artifacts_dir / job_id / "drift_report.html"
    summary = monitor.run_drift_report(reference_df, current_df, output_html_path)

    retrain_recommended = summary["dataset_drift_detected"] or summary["drift_share"] >= threshold
    retrain_job_id = None
    retrain_message = None

    if retrain_recommended:
        retrain_message = (
            f"Drift share {summary['drift_share']*100:.1f}% exceeds threshold — model retraining recommended."
        )
        if auto_retrain:
            retrain_job_id = new_id("job_")
            background_tasks.add_task(
                _trigger_retrain,
                job_id,
                entry["dataset_id"],
                entry["target_column"],
                retrain_job_id,
                n_trials,
            )
            retrain_message += f" Auto-retrain started: {retrain_job_id}"

    logger.info("[%s] Drift check complete: %s", job_id, summary)

    return DriftSummary(
        job_id=job_id,
        retrain_recommended=retrain_recommended,
        retrain_job_id=retrain_job_id,
        retrain_message=retrain_message,
        **summary,
    )


@router.get("/drift/{job_id}/report")
async def get_drift_report(job_id: str) -> FileResponse:
    settings = get_settings()
    report_path = settings.artifacts_dir / job_id / "drift_report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No drift report generated yet for this job.")
    return FileResponse(report_path, media_type="text/html")
