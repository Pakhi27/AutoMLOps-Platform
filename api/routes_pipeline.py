"""Pipeline execution endpoints: kick off a full AutoML run and poll status."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.schemas import JobRecord, JobStatus, PipelineRunAccepted, PipelineRunRequest
from app.services.modality.dataset_analysis import resolve_dataset_path
from app.services.modality.detector import load_modality_metadata
from app.services.modality.registry import get_modality_registry
from app.services.model_registry import get_model_registry
from app.store.job_store import get_job_store
from app.utils.io_utils import new_id

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = get_logger(__name__)


def _resolve_dataset_path(settings, dataset_id: str) -> Path:
    path = resolve_dataset_path(settings.upload_dir, dataset_id)
    return path or (settings.upload_dir / f"{dataset_id}.csv")


def _execute_job(job_id: str, request: PipelineRunRequest) -> None:
    store = get_job_store()
    settings = get_settings()
    store.update(job_id, status=JobStatus.running.value)

    meta = load_modality_metadata(settings.upload_dir, request.dataset_id)
    modality = (request.modality.value if request.modality else None) or meta.get("modality", "tabular")
    pipeline_type = meta.get("pipeline_type", "tabular_automl")
    store.update(job_id, modality=modality, pipeline_type=pipeline_type)

    dataset_path = _resolve_dataset_path(settings, request.dataset_id)
    if not dataset_path.exists():
        store.update(job_id, status=JobStatus.failed.value, error=f"Dataset file not found: {dataset_path}")
        return

    if request.text_column:
        meta["text_column"] = request.text_column
    if request.datetime_column:
        meta["datetime_column"] = request.datetime_column
    meta["dataset_id"] = request.dataset_id

    def progress(stage: str, message: str, pct: int) -> None:
        store.set_progress(job_id, stage, message, pct)

    cfg_model = settings.modeling
    test_size = request.test_size if request.test_size is not None else cfg_model.get("test_size", 0.2)

    try:
        pipeline = get_modality_registry().create(modality, progress_callback=progress)
        result = pipeline.run(
            job_id=job_id,
            dataset_path=str(dataset_path),
            target_column=request.target_column,
            metadata=meta,
            dataset_id=request.dataset_id,
            task_type=request.task_type.value if request.task_type else None,
            n_trials=request.n_trials,
            test_size=test_size,
            text_column=request.text_column,
            datetime_column=request.datetime_column,
        )
        store.update(
            job_id,
            status=JobStatus.success.value,
            result=result,
            task_type=result.get("task_type"),
            modality=result.get("modality", modality),
            pipeline_type=result.get("pipeline_type", pipeline_type),
        )
        # Register non-tabular models in local registry (tabular registers inside orchestrator)
        if result.get("pipeline_path") and modality != "tabular":
            get_model_registry().register(
                job_id,
                {
                    "job_id": job_id,
                    "dataset_id": request.dataset_id,
                    "target_column": request.target_column,
                    "task_type": result.get("task_type", "unknown"),
                    "model_name": result.get("model_name", "unknown"),
                    "metrics": result.get("metrics", {}),
                    "pipeline_path": result["pipeline_path"],
                    "modality": result.get("modality", modality),
                    "pipeline_type": result.get("pipeline_type", pipeline_type),
                    "text_column": result.get("text_column"),
                    "datetime_column": result.get("datetime_column"),
                    "label_classes": result.get("label_classes"),
                },
            )
        logger.info("[%s] %s pipeline completed", job_id, modality)
    except Exception as exc:
        logger.exception("[%s] Pipeline run failed", job_id)
        store.update(job_id, status=JobStatus.failed.value, error=str(exc))


@router.post("/run", response_model=PipelineRunAccepted, status_code=202)
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks) -> PipelineRunAccepted:
    settings = get_settings()
    dataset_path = _resolve_dataset_path(settings, request.dataset_id)
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{request.dataset_id}' not found.")

    job_id = new_id("job_")
    store = get_job_store()
    store.create(job_id, request.dataset_id, request.target_column)

    background_tasks.add_task(_execute_job, job_id, request)

    meta = {}
    try:
        from app.services.modality.detector import load_modality_metadata
        meta = load_modality_metadata(settings.upload_dir, request.dataset_id)
    except Exception:
        pass

    return PipelineRunAccepted(
        job_id=job_id,
        status=JobStatus.pending,
        message=(
            f"Pipeline accepted ({meta.get('modality', 'tabular')} / {meta.get('pipeline_type', 'tabular_automl')}). "
            "Poll GET /pipeline/jobs/{job_id} for status."
        ),
        modality=meta.get("modality"),
        pipeline_type=meta.get("pipeline_type"),
    )


@router.get("/jobs", response_model=list[JobRecord])
async def list_jobs() -> list[JobRecord]:
    store = get_job_store()
    return [JobRecord(**record) for record in store.list_all()]


@router.get("/jobs/{job_id}", response_model=JobRecord)
async def get_job(job_id: str) -> JobRecord:
    store = get_job_store()
    record = store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobRecord(**record)


@router.get("/dashboard")
async def dashboard_stats() -> dict:
    """Aggregate stats for the platform dashboard."""
    from app.services.model_registry import get_model_registry

    store = get_job_store()
    registry = get_model_registry().list_all()
    jobs = store.list_all()

    datasets = {j.get("dataset_id") for j in jobs if j.get("dataset_id")}
    for entry in registry.values():
        if entry.get("dataset_id"):
            datasets.add(entry["dataset_id"])

    best_accuracy = 0.0
    best_metric_label = "—"
    latest_model = "—"
    latest_job_id = None

    for job_id, entry in registry.items():
        metrics = entry.get("metrics") or {}
        for key in ("accuracy", "roc_auc", "r2", "f1_weighted"):
            val = metrics.get(key)
            if isinstance(val, (int, float)) and val > best_accuracy:
                best_accuracy = float(val)
                best_metric_label = key.replace("_", " ").upper()

    if registry:
        latest_job_id = list(registry.keys())[-1]
        latest_model = registry[latest_job_id].get("model_name", "—").replace("_", " ")

    successful = sum(1 for j in jobs if j.get("status") == "success")
    advisor_suggestions = sum(
        1 for j in jobs
        if j.get("status") == "success" and (j.get("result") or {}).get("advisor_report")
    )

    history = []
    for job in reversed(jobs[-12:]):
        result = job.get("result") or {}
        metrics = result.get("metrics") or {}
        score = metrics.get("accuracy") or metrics.get("roc_auc") or metrics.get("r2")
        history.append({
            "job_id": job.get("job_id"),
            "dataset_id": job.get("dataset_id"),
            "target_column": job.get("target_column"),
            "status": job.get("status"),
            "model_name": result.get("model_name"),
            "score": round(float(score), 4) if isinstance(score, (int, float)) else None,
            "created_at": job.get("created_at"),
        })

    return {
        "models_trained": len(registry),
        "datasets_uploaded": len(datasets),
        "best_accuracy": round(best_accuracy * 100, 1) if best_accuracy <= 1 else round(best_accuracy, 1),
        "best_metric_label": best_metric_label,
        "latest_model": latest_model,
        "latest_job_id": latest_job_id,
        "active_drift_checks": len(registry),
        "ai_suggestions": advisor_suggestions or successful,
        "successful_runs": successful,
        "job_history": history,
    }
