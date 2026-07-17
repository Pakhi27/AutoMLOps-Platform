"""Local JSON-backed model registry.

Complements MLflow's own Model Registry: MLflow is the system of record for
experiment history and versioned model artifacts, while this lightweight
registry gives the API a fast way to resolve `job_id -> pipeline.joblib path
+ metadata` without querying the MLflow backend on every prediction request.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.config import get_settings
from app.utils.io_utils import load_json, save_json

_lock = threading.Lock()


class ModelRegistry:
    def __init__(self) -> None:
        self.settings = get_settings()

    def register(self, job_id: str, entry: dict[str, Any]) -> None:
        with _lock:
            data = load_json(self.settings.registry_file, default={})
            entry["registered_at"] = datetime.now(timezone.utc).isoformat()
            data[job_id] = entry
            save_json(self.settings.registry_file, data)

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        data = load_json(self.settings.registry_file, default={})
        return data.get(job_id)

    def list_all(self) -> dict[str, Any]:
        return load_json(self.settings.registry_file, default={})


_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
