"""Data-derived dataset context for advisor RAG — no hardcoded business domains."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _split_column_tokens(columns: list[str]) -> set[str]:
    out: set[str] = set()
    for col in columns:
        out |= _tokens(col.replace("_", " "))
    return out


def row_size_bucket(n_rows: int) -> str:
    if n_rows < 500:
        return "small"
    if n_rows < 10_000:
        return "medium"
    if n_rows < 500_000:
        return "large"
    return "xlarge"


def build_dataset_signature(columns: list[str], target_column: str | None = None) -> str:
    features = sorted(c for c in columns if c != target_column)
    return "|".join(features).lower()


@dataclass
class DatasetContext:
    """Structural fingerprint used for RAG retrieval and run comparison."""

    task_type: str = "classification"
    target_column: str = ""
    feature_columns: list[str] = field(default_factory=list)
    column_tokens: set[str] = field(default_factory=set)
    dataset_signature: str = ""
    n_rows: int = 0
    n_columns: int = 0
    row_bucket: str = "small"
    is_imbalanced: bool = False
    has_missing: bool = False
    has_datetime: bool = False
    dataset_id: str = ""

    def to_rag_context(self, *, current_job_id: str | None = None, mode: str = "pre_train") -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "column_tokens": list(self.column_tokens),
            "dataset_signature": self.dataset_signature,
            "n_rows": self.n_rows,
            "row_bucket": self.row_bucket,
            "is_imbalanced": self.is_imbalanced,
            "has_missing": self.has_missing,
            "has_datetime": self.has_datetime,
            "dataset_id": self.dataset_id,
            "current_job_id": current_job_id,
            "mode": mode,
        }


def build_dataset_context(
    profile: dict[str, Any],
    target_column: str | None = None,
    *,
    dataset_id: str = "",
    registry_entry: dict[str, Any] | None = None,
) -> DatasetContext:
    numeric = list(profile.get("numeric_columns") or [])
    categorical = list(profile.get("categorical_columns") or [])
    datetime_cols = list(profile.get("datetime_columns") or [])
    target = target_column or (registry_entry or {}).get("target_column") or ""

    if registry_entry and registry_entry.get("feature_columns"):
        features = [c for c in registry_entry["feature_columns"] if c != target]
    else:
        features = [c for c in numeric + categorical if c != target]

    all_columns = features + ([target] if target else [])
    ta = profile.get("target_analysis") or {}
    task_type = ta.get("type") or (registry_entry or {}).get("task_type") or "classification"

    missing = profile.get("missing_values") or {}
    has_missing = any(v.get("n_missing", 0) > 0 for v in missing.values())

    n_rows = int(profile.get("n_rows") or (registry_entry or {}).get("n_rows") or 0)

    return DatasetContext(
        task_type=task_type,
        target_column=target,
        feature_columns=features,
        column_tokens=_split_column_tokens(all_columns),
        dataset_signature=build_dataset_signature(all_columns, target),
        n_rows=n_rows,
        n_columns=int(profile.get("n_columns") or len(all_columns)),
        row_bucket=row_size_bucket(n_rows),
        is_imbalanced=bool(ta.get("is_imbalanced")),
        has_missing=has_missing,
        has_datetime=bool(datetime_cols),
        dataset_id=dataset_id or (registry_entry or {}).get("dataset_id") or "",
    )
