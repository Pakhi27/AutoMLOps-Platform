"""Prediction endpoint: single-row, batch CSV, and feature importance."""
from __future__ import annotations

from functools import lru_cache
import joblib
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.schemas import BatchPredictResponse, FeatureImportanceResponse, PredictRequest, PredictResponse
from app.services.explainability import ExplainabilityService
from app.services.feature_importance import load_or_compute
from app.services.modality.image_utils import is_image_file, resolve_image_path
from app.services.modality.predict_handlers import load_artifact, predict_modality
from app.services.model_registry import get_model_registry
from app.utils.io_utils import read_csv_safely

router = APIRouter(prefix="/predict", tags=["predict"])
logger = get_logger(__name__)


@router.get("/preview-image")
async def preview_image(path: str = Query(..., description="Local image file path from image_path predict field")) -> FileResponse:
    """Serve a local image for UI preview during image predict."""
    resolved = resolve_image_path(path)
    if not is_image_file(resolved):
        raise HTTPException(status_code=404, detail=f"Image not found or unsupported format: {path}")
    ext = resolved.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    return FileResponse(resolved, media_type=media_map.get(ext, "application/octet-stream"))


@lru_cache(maxsize=16)
def _load_pipeline(pipeline_path: str):
    return load_artifact(pipeline_path)


def _run_predict(pipeline, entry: dict, df: pd.DataFrame, explain: bool = False) -> dict:
    modality = entry.get("modality", "tabular")
    if modality != "tabular" or (isinstance(pipeline, dict) and "text_column" in pipeline):
        return predict_modality(entry, pipeline, df, explain=explain)

    feature_columns = entry.get("feature_columns")
    if not feature_columns:
        raise ValueError(
            "Tabular model missing feature_columns. Retrain or use a tabular job_id."
        )
    missing_cols = [c for c in feature_columns if c not in df.columns]
    for c in missing_cols:
        df[c] = None
    df = df[feature_columns]

    raw_preds = pipeline.predict(df)
    label_classes = entry.get("label_classes")
    if label_classes:
        predictions = [label_classes[int(p)] for p in raw_preds]
    else:
        predictions = [float(p) for p in raw_preds]

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(df).tolist()
        except Exception:
            pass

    explanations = None
    if explain:
        try:
            settings = get_settings()
            job_id = entry["job_id"]
            reference_df = read_csv_safely(settings.reference_dir / f"{job_id}.csv")
            explainer_service = ExplainabilityService(sample_size=settings.explainability.get("shap_sample_size", 200))
            if entry["task_type"] == "classification":
                predict_fn = lambda data: pipeline.predict_proba(data)  # noqa: E731
            else:
                predict_fn = lambda data: pipeline.predict(data)  # noqa: E731
            explainer_service.build_explainer(predict_fn, reference_df[feature_columns])
            explanations = explainer_service.explain_rows(df)
        except Exception as exc:
            logger.warning("SHAP explanation failed: %s", exc)

    return {"predictions": predictions, "probabilities": probabilities, "explanations": explanations}


@router.post("/{job_id}", response_model=PredictResponse)
async def predict(job_id: str, request: PredictRequest) -> PredictResponse:
    registry = get_model_registry()
    entry = registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No trained model found for job '{job_id}'.")

    if not request.records:
        raise HTTPException(status_code=400, detail="`records` must contain at least one row.")

    pipeline = _load_pipeline(entry["pipeline_path"])
    df = pd.DataFrame(request.records)

    try:
        result = _run_predict(pipeline, entry, df, explain=request.explain)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        job_id=job_id,
        model_name=entry.get("model_name", "unknown"),
        task_type=entry.get("task_type", "unknown"),
        modality=entry.get("modality"),
        label_classes=result.get("label_classes") or entry.get("label_classes"),
        image_rows=result.get("image_rows"),
        **{k: v for k, v in result.items() if k not in ("label_classes", "image_rows")},
    )


@router.post("/{job_id}/batch", response_model=BatchPredictResponse)
async def batch_predict(job_id: str, file: UploadFile = File(...)) -> BatchPredictResponse:
    registry = get_model_registry()
    entry = registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No trained model found for job '{job_id}'.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported for batch predict.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    settings = get_settings()
    job_dir = settings.artifacts_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / "batch_input.csv"
    output_path = job_dir / "batch_predictions.csv"

    with open(input_path, "wb") as f:
        f.write(contents)

    df = read_csv_safely(input_path)
    pipeline = _load_pipeline(entry["pipeline_path"])

    try:
        result = _run_predict(pipeline, entry, df.copy(), explain=False)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Batch prediction failed: {exc}") from exc

    out_df = df.copy()
    out_df["prediction"] = result["predictions"]
    if result.get("probabilities") and len(result["probabilities"][0]) == 2:
        out_df["probability"] = [p[1] for p in result["probabilities"]]
    out_df.to_csv(output_path, index=False)

    preview = out_df.head(10).to_dict(orient="records")
    return BatchPredictResponse(
        job_id=job_id,
        model_name=entry.get("model_name", "unknown"),
        task_type=entry.get("task_type", "unknown"),
        n_rows=len(out_df),
        download_path=f"/predict/{job_id}/batch/download",
        preview=preview,
    )


@router.get("/{job_id}/batch/download")
async def download_batch_predictions(job_id: str) -> FileResponse:
    settings = get_settings()
    output_path = settings.artifacts_dir / job_id / "batch_predictions.csv"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="No batch predictions found. Run POST /predict/{job_id}/batch first.")
    return FileResponse(output_path, media_type="text/csv", filename=f"predictions_{job_id}.csv")


@router.get("/{job_id}/feature-importance", response_model=FeatureImportanceResponse)
async def feature_importance(job_id: str) -> FeatureImportanceResponse:
    try:
        features = load_or_compute(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FeatureImportanceResponse(job_id=job_id, features=features)
