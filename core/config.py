"""Application configuration.

Loads defaults from config/config.yaml and allows overrides via environment
variables (useful for Docker / CI where paths and the MLflow tracking URI
need to change without editing the YAML file).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings:
    """Thin typed wrapper around the YAML config + env var overrides."""

    def __init__(self) -> None:
        raw = _load_yaml(CONFIG_PATH)

        paths = raw.get("paths", {})
        self.upload_dir = BASE_DIR / paths.get("upload_dir", "data/uploads")
        self.reference_dir = BASE_DIR / paths.get("reference_dir", "data/reference")
        self.artifacts_dir = BASE_DIR / paths.get("artifacts_dir", "artifacts")
        self.registry_file = BASE_DIR / paths.get("registry_file", "artifacts/registry.json")
        self.job_store_file = BASE_DIR / paths.get("job_store_file", "app/store/jobs.json")

        mlflow_cfg = raw.get("mlflow", {})
        self.mlflow_tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", mlflow_cfg.get("tracking_uri", "file:./mlruns")
        )
        self.mlflow_experiment_name = os.getenv(
            "MLFLOW_EXPERIMENT_NAME", mlflow_cfg.get("experiment_name", "automl-platform")
        )

        self.cleaning = raw.get("cleaning", {})
        self.outliers = raw.get("outliers", {})
        self.feature_engineering = raw.get("feature_engineering", {})
        self.modeling = raw.get("modeling", {})
        self.tuning = raw.get("tuning", {})
        self.explainability = raw.get("explainability", {})
        self.drift = raw.get("drift", {})
        self.leakage = raw.get("leakage", {})
        self.feature_selection = raw.get("feature_selection", {})
        self.mle = raw.get("mle", {})
        self.agent = raw.get("agent", {})
        self.modalities = raw.get("modalities", {})

        for d in (self.upload_dir, self.reference_dir, self.artifacts_dir, self.job_store_file.parent):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
