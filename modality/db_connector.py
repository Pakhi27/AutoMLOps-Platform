"""Database connector — import SQL table as dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.core.config import get_settings
from app.services.modality.detector import DataModalityDetector, save_modality_metadata
from app.utils.io_utils import new_id


def connect_and_export(
    connection_url: str,
    table: str,
    query: Optional[str] = None,
    schema: Optional[str] = None,
    limit: int = 100_000,
) -> dict[str, Any]:
    """Read from SQL database and save as dataset CSV + metadata."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise ImportError("Install sqlalchemy: pip install sqlalchemy") from exc

    settings = get_settings()
    engine = create_engine(connection_url)
    if query:
        sql = text(query)
    else:
        tbl = f"{schema}.{table}" if schema else table
        sql = text(f"SELECT * FROM {tbl} LIMIT :lim")
    df = pd.read_sql(sql, engine, params={"lim": limit} if not query else None)
    if df.empty:
        raise ValueError("Query returned no rows.")

    dataset_id = new_id("ds_")
    dest = settings.upload_dir / f"{dataset_id}.csv"
    df.to_csv(dest, index=False)

    detector = DataModalityDetector()
    meta = detector.detect_dataframe(df, filename=f"{table}.csv")
    meta["dataset_id"] = dataset_id
    meta["source"] = "database"
    meta["connection"] = connection_url.split("@")[-1] if "@" in connection_url else "database"
    meta["table"] = table
    save_modality_metadata(settings.upload_dir, dataset_id, meta)

    return {
        "dataset_id": dataset_id,
        "filename": f"{table}.csv",
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": df.columns.tolist(),
        **meta,
    }
