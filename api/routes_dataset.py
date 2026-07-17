"""Dataset upload & profiling endpoints."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.schemas import DatasetUploadResponse, EDAResponse, ProfileResponse
from app.services.data_profiler import DataProfiler
from app.services.eda_charts import EDAChartGenerator
from app.services.eda_service import EDAService
from app.services.modality.detector import DataModalityDetector, save_modality_metadata
from app.utils.io_utils import new_id, read_csv_safely

router = APIRouter(prefix="/datasets", tags=["datasets"])
logger = get_logger(__name__)


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are currently supported.")

    settings = get_settings()
    dataset_id = new_id("ds_")
    dest_path = settings.upload_dir / f"{dataset_id}.csv"

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with open(dest_path, "wb") as f:
        f.write(contents)

    try:
        df = read_csv_safely(dest_path)
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if df.empty:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded CSV has no rows.")

    logger.info("Uploaded dataset %s (%s) -> %d rows, %d cols", dataset_id, file.filename, *df.shape)

    meta = DataModalityDetector().detect_dataframe(df, file.filename or dest_path.name)
    meta["dataset_id"] = dataset_id
    save_modality_metadata(settings.upload_dir, dataset_id, meta)

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        filename=file.filename,
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=df.columns.tolist(),
        modality=meta.get("modality", "tabular"),
        pipeline_type=meta.get("pipeline_type", "tabular_automl"),
        detection_reason=meta.get("detection_reason"),
        suggested_targets=meta.get("suggested_targets", []),
        text_column=meta.get("text_column"),
        datetime_column=meta.get("datetime_column"),
    )


@router.get("/{dataset_id}/profile", response_model=ProfileResponse)
async def profile_dataset(dataset_id: str) -> ProfileResponse:
    settings = get_settings()
    path = settings.upload_dir / f"{dataset_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    df = read_csv_safely(path)
    profile = DataProfiler().profile(df)
    return ProfileResponse(dataset_id=dataset_id, profile=profile)


@router.get("/{dataset_id}/eda", response_model=EDAResponse)
async def eda_dataset(dataset_id: str, target_column: str = Query(...)) -> EDAResponse:
    settings = get_settings()
    path = settings.upload_dir / f"{dataset_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    df = read_csv_safely(path)
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not in dataset.")

    eda = EDAService().analyze(df, target_column)
    charts = EDAChartGenerator(dataset_id, target_column).generate_all(df, eda)
    eda["charts"] = charts
    return EDAResponse(dataset_id=dataset_id, target_column=target_column, eda=eda)


@router.get("/{dataset_id}/eda/chart/{target_column}/{chart_file}")
async def get_eda_chart(dataset_id: str, target_column: str, chart_file: str) -> FileResponse:
    settings = get_settings()
    if ".." in chart_file or "/" in chart_file:
        raise HTTPException(status_code=400, detail="Invalid chart path.")
    path = settings.artifacts_dir / "eda" / dataset_id / target_column / chart_file
    if not path.exists() or not path.suffix == ".png":
        raise HTTPException(status_code=404, detail="Chart not found. Run EDA first.")
    return FileResponse(path, media_type="image/png")
