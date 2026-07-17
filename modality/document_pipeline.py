"""PDF / document pipeline — extract text then classify or index for RAG."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.core.config import get_settings
from app.services.modality.base import BaseModalityPipeline
from app.services.modality.text_pipeline import TextModalityPipeline
from app.utils.io_utils import read_csv_safely, new_id


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf: pip install pypdf") from exc
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages[:50]]
    return "\n".join(pages).strip()


class DocumentModalityPipeline(BaseModalityPipeline):
    modality = "documents"
    pipeline_type = "document_classification"

    def run(
        self,
        job_id: str,
        dataset_path: str,
        target_column: str,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        t0 = time.time()
        settings = get_settings()
        path = Path(dataset_path)

        self.report("preprocessing", "Extracting text from documents", 15)
        if path.suffix.lower() == ".pdf":
            text = _extract_pdf_text(path)
            df = pd.DataFrame({"text": [text], target_column: [metadata.get("default_label", "document")]})
            csv_path = settings.upload_dir / f"{metadata.get('dataset_id', new_id('ds_'))}_extracted.csv"
            df.to_csv(csv_path, index=False)
            dataset_path = str(csv_path)
        elif path.suffix.lower() in {".csv", ".xlsx"}:
            df = read_csv_safely(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
            text_col = metadata.get("text_column")
            if text_col and text_col in df.columns:
                pass
            else:
                obj_cols = df.select_dtypes(include=["object"]).columns
                if len(obj_cols):
                    metadata["text_column"] = str(obj_cols[0])
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            df = pd.DataFrame({"text": [text], target_column: ["document"]})
            csv_path = settings.upload_dir / f"{new_id('ds_')}_doc.csv"
            df.to_csv(csv_path, index=False)
            dataset_path = str(csv_path)
            metadata["text_column"] = "text"

        self.report("preprocessing", "Chunking for RAG index", 35)
        rag_chunks = self._build_rag_chunks(read_csv_safely(Path(dataset_path)), metadata.get("text_column", "text"))
        out_dir = settings.artifacts_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "document_rag_chunks.json").write_text(json.dumps(rag_chunks[:200], indent=2), encoding="utf-8")

        if target_column in read_csv_safely(Path(dataset_path)).columns:
            self.report("training", "Document classification", 55)
            result = TextModalityPipeline(progress_callback=self._progress).run(
                job_id=job_id,
                dataset_path=dataset_path,
                target_column=target_column,
                metadata={**metadata, "text_column": metadata.get("text_column", "text")},
                **kwargs,
            )
        else:
            result = {
                "job_id": job_id,
                "task_type": "indexing",
                "model_name": "rag_index",
                "metrics": {"chunks_indexed": len(rag_chunks)},
            }

        result["modality"] = self.modality
        result["pipeline_type"] = self.pipeline_type
        result["rag_chunks_path"] = str(out_dir / "document_rag_chunks.json")
        result["drift_notes"] = ["Topic drift", "Vocabulary shift", "Retrieval quality decay"]
        result["training_seconds"] = round(time.time() - t0, 1)
        return result

    @staticmethod
    def _build_rag_chunks(df: pd.DataFrame, text_col: str, chunk_size: int = 500) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        if text_col not in df.columns:
            return chunks
        for i, row in df.head(100).iterrows():
            text = str(row[text_col])
            for j in range(0, len(text), chunk_size):
                piece = text[j : j + chunk_size].strip()
                if piece:
                    chunks.append({"doc_id": str(i), "chunk_id": f"{i}_{j}", "text": piece})
        return chunks
