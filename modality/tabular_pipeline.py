"""Tabular pipeline — delegates to existing PipelineOrchestrator."""
from __future__ import annotations

from typing import Any, Optional

from app.services.modality.base import BaseModalityPipeline
from app.services.pipeline_orchestrator import PipelineOrchestrator


class TabularModalityPipeline(BaseModalityPipeline):
    modality = "tabular"
    pipeline_type = "tabular_automl"

    def run(
        self,
        job_id: str,
        dataset_path: str,
        target_column: str,
        metadata: dict[str, Any],
        dataset_id: str = "",
        task_type: Optional[str] = None,
        n_trials: Optional[int] = None,
        test_size: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(
            job_id=job_id,
            dataset_id=dataset_id or metadata.get("dataset_id", ""),
            target_column=target_column,
            task_type=task_type,
            n_trials=n_trials,
            test_size=test_size,
        )
        result["modality"] = self.modality
        result["pipeline_type"] = self.pipeline_type
        return result
