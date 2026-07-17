"""Multi-modal upload, connect, and capability endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.logging_config import get_logger
from app.schemas.schemas import DatabaseConnectRequest, DatasetUploadResponse, ModalityInfoResponse
from app.services.modality.db_connector import connect_and_export
from app.services.modality.detector import DataModalityDetector, load_modality_metadata, save_modality_metadata
from app.services.modality.registry import get_modality_registry
from app.utils.io_utils import new_id, read_csv_safely

router = APIRouter(prefix="/multimodal", tags=["multimodal"])
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {
    ".csv", ".tsv", ".xlsx", ".xls", ".pdf", ".txt", ".jsonl", ".zip",
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp",
}


@router.get("/capabilities")
async def list_capabilities() -> dict:
    """Supported data modalities with preprocessing, models, metrics, explainability, drift."""
    registry = get_modality_registry()
    return {
        "flow": [
            "Upload / Connect Data",
            "Auto-detect Data Type",
            "Select Pipeline Automatically",
            "Run Preprocessing",
            "Train Suitable Models",
            "Evaluate with Correct Metrics",
            "Generate Explainability",
            "Monitor Drift",
            "Create AI Advisor Report",
        ],
        "modalities": registry.capabilities(),
    }


@router.get("/datasets/{dataset_id}/modality", response_model=ModalityInfoResponse)
async def get_modality(dataset_id: str) -> ModalityInfoResponse:
    settings = get_settings()
    meta = load_modality_metadata(settings.upload_dir, dataset_id)
    if not any(settings.upload_dir.glob(f"{dataset_id}*")):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return ModalityInfoResponse(dataset_id=dataset_id, **{k: v for k, v in meta.items() if k != "dataset_id"})


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_multimodal(file: UploadFile = File(...)) -> DatasetUploadResponse:
    """Upload CSV, Excel, PDF, text, images (ZIP), and auto-detect modality."""
    settings = get_settings()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    dataset_id = new_id("ds_")
    dest = settings.upload_dir / f"{dataset_id}{ext}"
    with open(dest, "wb") as f:
        f.write(contents)

    detector = DataModalityDetector()
    meta = detector.detect_file(dest, file.filename or dest.name)
    meta["dataset_id"] = dataset_id
    save_modality_metadata(settings.upload_dir, dataset_id, meta)

    n_rows, n_columns, columns = 0, 0, []
    if ext in {".csv", ".tsv", ".xlsx", ".xls"}:
        try:
            if ext in {".xlsx", ".xls"}:
                import pandas as pd
                df = pd.read_excel(dest)
            elif ext == ".tsv":
                import pandas as pd
                df = pd.read_csv(dest, sep="\t")
            else:
                df = read_csv_safely(dest)
            n_rows, n_columns = len(df), len(df.columns)
            columns = df.columns.tolist()
            # Re-detect from dataframe for better accuracy
            meta = detector.detect_dataframe(df, file.filename or dest.name)
            meta["dataset_id"] = dataset_id
            save_modality_metadata(settings.upload_dir, dataset_id, meta)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}") from exc
    elif ext == ".pdf":
        n_rows, n_columns, columns = 1, 2, ["document_text", "label"]
        meta.setdefault("suggested_targets", ["label"])
        meta.setdefault("text_column", "document_text")
    elif ext in {".txt", ".md"}:
        n_rows, n_columns, columns = 1, 2, ["text", "label"]
        meta.setdefault("suggested_targets", ["label"])
        meta.setdefault("text_column", "text")
    elif ext in {".zip", ".jpg", ".jpeg", ".png"}:
        n_rows, n_columns, columns = 1, 2, ["image_path", "label"]
        meta.setdefault("suggested_targets", ["label"])

    # Also save as .csv alias path for tabular if csv
    if ext == ".csv":
        pass
    elif ext in {".xlsx", ".xls", ".tsv"} and n_rows:
        csv_dest = settings.upload_dir / f"{dataset_id}.csv"
        import pandas as pd
        df = pd.read_excel(dest) if ext in {".xlsx", ".xls"} else pd.read_csv(dest, sep="\t")
        df.to_csv(csv_dest, index=False)

    logger.info(
        "Uploaded %s (%s) modality=%s pipeline=%s",
        dataset_id, file.filename, meta.get("modality"), meta.get("pipeline_type"),
    )

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        filename=file.filename or dest.name,
        n_rows=n_rows,
        n_columns=n_columns,
        columns=columns,
        modality=meta.get("modality", "tabular"),
        pipeline_type=meta.get("pipeline_type", "tabular_automl"),
        detection_reason=meta.get("detection_reason"),
        suggested_targets=meta.get("suggested_targets", []),
        text_column=meta.get("text_column"),
        datetime_column=meta.get("datetime_column"),
    )


@router.post("/connect-database", response_model=DatasetUploadResponse)
async def connect_database(request: DatabaseConnectRequest) -> DatasetUploadResponse:
    """Connect to SQL database, export table to dataset, auto-detect modality."""
    try:
        result = connect_and_export(
            connection_url=request.connection_url,
            table=request.table,
            query=request.query,
            schema=request.schema,
            limit=request.limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DatasetUploadResponse(
        dataset_id=result["dataset_id"],
        filename=result["filename"],
        n_rows=result["n_rows"],
        n_columns=result["n_columns"],
        columns=result["columns"],
        modality=result.get("modality", "tabular"),
        pipeline_type=result.get("pipeline_type", "tabular_automl"),
        detection_reason=result.get("detection_reason"),
        suggested_targets=result.get("suggested_targets", []),
        text_column=result.get("text_column"),
        datetime_column=result.get("datetime_column"),
    )
