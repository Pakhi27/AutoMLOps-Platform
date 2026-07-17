"""ML Intelligence endpoints — quality, leakage, review, counterfactuals, model cards."""
from __future__ import annotations

import json

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.schemas import (
    ActiveLearningResponse,
    BusinessInsightsResponse,
    CounterfactualRequest,
    CounterfactualResponse,
    DatasetQualityResponse,
    ExperimentCompareResponse,
    FeatureDriftResponse,
    LeakageReportResponse,
    ModelCardResponse,
    ModelReviewResponse,
)
from app.services.active_learning import ActiveLearningService
from app.services.business_insights import BusinessInsightGenerator
from app.services.counterfactuals import CounterfactualExplainer
from app.services.dataset_quality import DatasetQualityScorer
from app.services.drift_monitor import DriftMonitor
from app.services.eda_service import EDAService
from app.services.experiment_comparator import ExperimentComparator
from app.services.modality.dataset_analysis import (
    image_leakage_report,
    load_tabular_dataframe,
    resolve_dataset_path,
    score_image_quality,
    summarize_image_dataset,
)
from app.services.modality.detector import load_modality_metadata
from app.services.leakage_detector import LeakageDetector
from app.services.model_card_generator import ModelCardGenerator
from app.services.model_registry import get_model_registry
from app.services.model_reviewer import ModelReviewer
from app.utils.io_utils import read_csv_safely

router = APIRouter(prefix="/mle", tags=["mle-intelligence"])
logger = get_logger(__name__)


def _load_job_artifact(job_id: str, name: str) -> dict | None:
    path = get_settings().artifacts_dir / job_id / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _get_registry_entry(job_id: str) -> dict:
    entry = get_model_registry().get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return entry


@router.get("/datasets/{dataset_id}/leakage", response_model=LeakageReportResponse)
async def check_leakage(dataset_id: str, target_column: str = Query(...)) -> LeakageReportResponse:
    settings = get_settings()
    meta = load_modality_metadata(settings.upload_dir, dataset_id)
    modality = meta.get("modality", "tabular")

    if modality == "image":
        if resolve_dataset_path(settings.upload_dir, dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        summary = summarize_image_dataset(settings.upload_dir, dataset_id)
        report = image_leakage_report(summary, target_column)
        return LeakageReportResponse(dataset_id=dataset_id, target_column=target_column, report=report)

    df = load_tabular_dataframe(settings.upload_dir, dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    cfg = settings.leakage
    report = LeakageDetector(
        target_corr_threshold=cfg.get("target_corr_threshold", 0.98),
        duplicate_corr_threshold=cfg.get("duplicate_corr_threshold", 0.999),
    ).detect(df, target_column)
    return LeakageReportResponse(dataset_id=dataset_id, target_column=target_column, report=report)


@router.get("/datasets/{dataset_id}/quality", response_model=DatasetQualityResponse)
async def dataset_quality(dataset_id: str, target_column: str = Query(...)) -> DatasetQualityResponse:
    settings = get_settings()
    meta = load_modality_metadata(settings.upload_dir, dataset_id)
    modality = meta.get("modality", "tabular")

    if modality == "image":
        if resolve_dataset_path(settings.upload_dir, dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        summary = summarize_image_dataset(settings.upload_dir, dataset_id)
        quality = score_image_quality(summary)
        return DatasetQualityResponse(dataset_id=dataset_id, target_column=target_column, quality=quality)

    df = load_tabular_dataframe(settings.upload_dir, dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    quality = DatasetQualityScorer().score(df, target_column)
    return DatasetQualityResponse(dataset_id=dataset_id, target_column=target_column, quality=quality)


@router.get("/jobs/{job_id}/review", response_model=ModelReviewResponse)
async def model_review(job_id: str) -> ModelReviewResponse:
    cached = _load_job_artifact(job_id, "model_review.json")
    if cached:
        return ModelReviewResponse(job_id=job_id, review=cached)
    entry = _get_registry_entry(job_id)
    review = ModelReviewer().review(
        job_id=job_id,
        model_name=entry.get("model_name", "unknown"),
        task_type=entry.get("task_type", "classification"),
        metrics=entry.get("metrics", {}),
        baseline_scores=entry.get("baseline_scores", {}),
        best_params=entry.get("best_params", {}),
        feature_importance=entry.get("top_features", []),
        n_train=entry.get("n_rows", 0),
        n_test=0,
        feature_selection=_load_job_artifact(job_id, "feature_selection.json"),
        leakage_report=_load_job_artifact(job_id, "leakage_report.json"),
    )
    return ModelReviewResponse(job_id=job_id, review=review)


@router.get("/jobs/{job_id}/business-insights", response_model=BusinessInsightsResponse)
async def business_insights(job_id: str) -> BusinessInsightsResponse:
    cached = _load_job_artifact(job_id, "business_insights.json")
    if cached:
        return BusinessInsightsResponse(job_id=job_id, insights=cached)
    entry = _get_registry_entry(job_id)
    settings = get_settings()
    df = load_tabular_dataframe(settings.upload_dir, entry["dataset_id"])
    eda = None
    if df is not None:
        eda = EDAService().analyze(df, entry["target_column"])
    insights = BusinessInsightGenerator().generate(
        target_column=entry["target_column"],
        task_type=entry.get("task_type", "classification"),
        feature_importance=entry.get("top_features", []),
        metrics=entry.get("metrics"),
        eda=eda,
    )
    return BusinessInsightsResponse(job_id=job_id, insights=insights)


@router.get("/jobs/{job_id}/model-card", response_model=ModelCardResponse)
async def get_model_card(job_id: str) -> ModelCardResponse:
    job_dir = get_settings().artifacts_dir / job_id
    md_path = job_dir / "model_card.md"
    json_path = job_dir / "model_card.json"
    if json_path.exists() and md_path.exists():
        card = json.loads(json_path.read_text(encoding="utf-8"))
        markdown = md_path.read_text(encoding="utf-8")
        return ModelCardResponse(job_id=job_id, card=card, markdown=markdown)
    entry = _get_registry_entry(job_id)
    card_data = ModelCardGenerator().generate(
        job_id=job_id,
        registry_entry=entry,
        model_review=_load_job_artifact(job_id, "model_review.json"),
        business_insights=_load_job_artifact(job_id, "business_insights.json"),
        feature_selection=_load_job_artifact(job_id, "feature_selection.json"),
        leakage_report=_load_job_artifact(job_id, "leakage_report.json"),
    )
    return ModelCardResponse(job_id=job_id, card=card_data["card"], markdown=card_data["markdown"])


@router.get("/jobs/{job_id}/model-card/download")
async def download_model_card(job_id: str, format: str = Query("md", pattern="^(md|json)$")):
    job_dir = get_settings().artifacts_dir / job_id
    path = job_dir / ("model_card.md" if format == "md" else "model_card.json")
    if not path.exists():
        await get_model_card(job_id)
        path = job_dir / ("model_card.md" if format == "md" else "model_card.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Model card not generated yet.")
    media = "text/markdown" if format == "md" else "application/json"
    return FileResponse(path, media_type=media, filename=f"model_card_{job_id}.{format}")


@router.get("/experiments/compare", response_model=ExperimentCompareResponse)
async def compare_experiments(
    job_a: str = Query(...),
    job_b: str = Query(...),
) -> ExperimentCompareResponse:
    try:
        comparison = ExperimentComparator().compare(job_a, job_b)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ExperimentCompareResponse(comparison=comparison)


@router.post("/jobs/{job_id}/counterfactual", response_model=CounterfactualResponse)
async def counterfactual(job_id: str, request: CounterfactualRequest) -> CounterfactualResponse:
    import joblib

    entry = _get_registry_entry(job_id)
    modality = entry.get("modality", "tabular")
    if modality != "tabular":
        raise HTTPException(
            status_code=422,
            detail=(
                f"Counterfactual explanations are only available for tabular models "
                f"(CSV with numeric/categorical features). This job is '{modality}'. "
                "Use SHAP / feature importance for tabular models, or class probabilities on image predict."
            ),
        )
    if not entry.get("feature_columns"):
        raise HTTPException(
            status_code=422,
            detail="This model has no feature_columns. Retrain a tabular job to use counterfactuals.",
        )

    pipeline = joblib.load(entry["pipeline_path"])
    try:
        result = CounterfactualExplainer().explain(pipeline, entry, request.record)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Counterfactual failed: {exc}") from exc
    return CounterfactualResponse(job_id=job_id, result=result)


@router.post("/jobs/{job_id}/active-learning", response_model=ActiveLearningResponse)
async def active_learning(job_id: str, file: UploadFile = File(...)) -> ActiveLearningResponse:
    import joblib

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV required.")
    entry = _get_registry_entry(job_id)
    contents = await file.read()
    settings = get_settings()
    tmp = settings.artifacts_dir / job_id / "active_learning_input.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(contents)
    df = read_csv_safely(tmp)
    pipeline = joblib.load(entry["pipeline_path"])
    threshold = settings.mle.get("active_learning_threshold", 0.55)
    result = ActiveLearningService(uncertainty_threshold=threshold).score_batch(pipeline, entry, df)
    out_path = settings.artifacts_dir / job_id / "active_learning.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return ActiveLearningResponse(job_id=job_id, result=result)


@router.post("/monitor/drift/{job_id}/features", response_model=FeatureDriftResponse)
async def feature_drift(job_id: str, file: UploadFile = File(...)) -> FeatureDriftResponse:
    entry = _get_registry_entry(job_id)
    ref_path = entry.get("reference_data_path")
    if not ref_path:
        raise HTTPException(status_code=404, detail="Reference data not found.")
    contents = await file.read()
    settings = get_settings()
    tmp = settings.artifacts_dir / job_id / "feature_drift_current.csv"
    tmp.write_bytes(contents)
    reference = read_csv_safely(ref_path)
    current = read_csv_safely(tmp)
    monitor = DriftMonitor(settings.drift.get("dataset_drift_share_threshold", 0.5))
    output_html = settings.artifacts_dir / job_id / "feature_drift_report.html"
    dataset_drift = monitor.run_drift_report(reference, current, output_html)
    feature_drift_list = dataset_drift.get("feature_drift") or monitor.get_feature_drift(reference, current)
    return FeatureDriftResponse(
        job_id=job_id,
        feature_drift=feature_drift_list,
        dataset_drift={
            k: v for k, v in dataset_drift.items()
            if k in ("dataset_drift_detected", "drift_share", "number_of_drifted_columns", "report_html_path")
        },
    )
