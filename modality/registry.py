"""Route jobs to the correct modality pipeline."""
from __future__ import annotations

from typing import Any, Callable, Optional

from app.services.modality.base import BaseModalityPipeline
from app.services.modality.document_pipeline import DocumentModalityPipeline
from app.services.modality.image_pipeline import ImageModalityPipeline
from app.services.modality.logs_pipeline import LogsModalityPipeline
from app.services.modality.tabular_pipeline import TabularModalityPipeline
from app.services.modality.text_pipeline import TextModalityPipeline
from app.services.modality.timeseries_pipeline import TimeSeriesModalityPipeline


class ModalityRegistry:
    def __init__(self) -> None:
        self._pipelines: dict[str, type[BaseModalityPipeline]] = {
            "tabular": TabularModalityPipeline,
            "text": TextModalityPipeline,
            "image": ImageModalityPipeline,
            "timeseries": TimeSeriesModalityPipeline,
            "logs": LogsModalityPipeline,
            "documents": DocumentModalityPipeline,
        }

    def get(self, modality: str) -> type[BaseModalityPipeline]:
        key = modality.lower().strip()
        if key not in self._pipelines:
            return TabularModalityPipeline
        return self._pipelines[key]

    def create(
        self,
        modality: str,
        progress_callback: Optional[Callable[[str, str, int], None]] = None,
    ) -> BaseModalityPipeline:
        return self.get(modality)(progress_callback=progress_callback)

    def capabilities(self) -> list[dict[str, Any]]:
        from app.services.modality.detector import _DRIFT, _EXPLAIN, _METRICS, _MODELS, _PREPROCESSING

        rows = []
        for key in self._pipelines:
            rows.append({
                "modality": key,
                "pipeline_type": self._pipelines[key].pipeline_type,
                "preprocessing": _PREPROCESSING.get(key, []),
                "models": _MODELS.get(key, []),
                "metrics": _METRICS.get(key, []),
                "explainability": _EXPLAIN.get(key, []),
                "drift": _DRIFT.get(key, []),
            })
        return rows


_registry: Optional[ModalityRegistry] = None


def get_modality_registry() -> ModalityRegistry:
    global _registry
    if _registry is None:
        _registry = ModalityRegistry()
    return _registry
