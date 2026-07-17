"""Base class for modality-specific training pipelines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class BaseModalityPipeline(ABC):
    modality: str = "base"
    pipeline_type: str = "base"

    def __init__(self, progress_callback: Optional[Callable[[str, str, int], None]] = None) -> None:
        self._progress = progress_callback or (lambda _s, _m, _p: None)

    def report(self, stage: str, message: str, pct: int) -> None:
        self._progress(stage, message, pct)

    @abstractmethod
    def run(
        self,
        job_id: str,
        dataset_path: str,
        target_column: str,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train and return a result dict compatible with JobRecord.result."""
