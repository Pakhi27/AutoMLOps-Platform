"""Auto-detect data modality from uploaded files or database tables."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.utils.io_utils import read_csv_safely

LOG_COL_PATTERNS = re.compile(
    r"(message|log|text|description|body|incident|ticket|error|stack|trace|assignment|severity|priority|level)",
    re.I,
)
TIME_COL_PATTERNS = re.compile(
    r"(^date$|_date$|^date_|timestamp|datetime|(^time$|_time$)|^period$|_period$|"
    r"^month$|_month$|^year$|_year$|^week$|_week$|signup_date|created_at|updated_at)",
    re.I,
)
TIME_COL_FALSE_POSITIVE = re.compile(
    r"(timezone|time_zone|location|coord|zone|country|city|name|monthly|tenure|charges|"
    r"charge|billing|duration|minute|second|hour|age|latency|runtime|uptime)",
    re.I,
)
TEXT_COL_PATTERNS = re.compile(r"(text|content|body|message|review|comment|description|ticket|summary|tweet)", re.I)
SENTIMENT_TARGET_PATTERNS = re.compile(r"(sentiment|label|class|category|polarity)", re.I)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
TABULAR_EXTS = {".csv", ".xlsx", ".xls", ".tsv"}
DOCUMENT_EXTS = {".pdf", ".docx", ".txt", ".md"}
TEXT_BULK_EXTS = {".txt", ".jsonl", ".json"}


class DataModalityDetector:
    """Classify uploads into modality + recommended pipeline."""

    def detect_file(self, path: Path, filename: str) -> dict[str, Any]:
        ext = Path(filename).suffix.lower()
        if ext in IMAGE_EXTS:
            return self._result("image", "image_classification", "Single image upload — use ZIP folder for training.")
        if ext == ".pdf":
            return self._result(
                "documents",
                "document_classification",
                "PDF document — text/table extraction + classification.",
                extra={"text_column": "document_text", "suggested_targets": ["label"]},
            )
        if ext in {".zip", ".tar", ".gz"}:
            return self._result(
                "image",
                "image_classification",
                "Archive — expected image folder per class.",
                extra={"suggested_targets": ["label"]},
            )
        if ext in TABULAR_EXTS:
            return self._detect_tabular_like(path, filename)
        if ext in {".txt", ".md"}:
            return self._result(
                "documents",
                "document_classification",
                "Plain-text document — extraction, chunking, and RAG index.",
                extra={"text_column": "text", "suggested_targets": ["label"]},
            )
        if ext in {".jsonl", ".json"}:
            return self._result(
                "text",
                "text_classification",
                "JSONL / JSON text corpus.",
                extra={"suggested_targets": ["label"]},
            )
        return self._result("tabular", "tabular_automl", f"Defaulting to tabular for extension {ext}.")

    def detect_dataframe(self, df: pd.DataFrame, filename: str = "data.csv") -> dict[str, Any]:
        return self._classify_dataframe(df, filename)

    def _detect_tabular_like(self, path: Path, filename: str) -> dict[str, Any]:
        try:
            if path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path, nrows=500)
            elif path.suffix.lower() == ".tsv":
                df = pd.read_csv(path, sep="\t", nrows=500)
            else:
                df = read_csv_safely(path)
                if len(df) > 500:
                    df = df.head(500)
        except Exception as exc:
            return self._result("tabular", "tabular_automl", f"Parse as tabular (warning: {exc})")
        return self._classify_dataframe(df, filename)

    def _classify_dataframe(self, df: pd.DataFrame, filename: str) -> dict[str, Any]:
        cols = list(df.columns)
        col_lower = [str(c).lower() for c in cols]

        # Logs / tickets
        log_hits = sum(1 for c in col_lower if LOG_COL_PATTERNS.search(c.replace(" ", "_")))
        if log_hits >= 2 or (log_hits >= 1 and any("assign" in c or "group" in c or "severity" in c for c in col_lower)):
            return self._result(
                "logs",
                "log_ticket_classification",
                "Structured logs/tickets detected (message + metadata columns).",
                extra={"text_column": self._pick_text_column(df), "suggested_targets": self._label_columns(df)},
            )

        # Text-heavy tabular (before time-series — avoids tweet_created/user_timezone false positives)
        text_cols = self._text_columns(df, cols)
        if text_cols:
            text_col = text_cols[0]
            avg_len = df[text_col].astype(str).str.len().mean()
            has_sentiment_target = any(SENTIMENT_TARGET_PATTERNS.search(str(c).lower()) for c in cols)
            if avg_len > 25 or TEXT_COL_PATTERNS.search(str(text_col).lower()) or has_sentiment_target:
                return self._result(
                    "text",
                    "text_classification",
                    "Long-text column detected — NLP classification/sentiment pipeline.",
                    extra={"text_column": text_col, "suggested_targets": self._label_columns(df, exclude=text_cols)},
                )

        # Time-series (strict: real parseable dates + forecastable numeric values)
        dt_cols = self._datetime_columns(df, cols)
        value_cols = self._timeseries_value_columns(df, cols)
        label_cols = self._label_columns(df)
        # Entity-level tabular data (e.g. churn) often has a signup/created date — prefer tabular when labels exist
        if dt_cols and value_cols and label_cols and not self._looks_like_time_index(df, dt_cols[0]):
            pass  # fall through to tabular below
        elif dt_cols and value_cols:
            return self._result(
                "timeseries",
                "timeseries_forecast",
                "Datetime + numeric columns — forecasting / anomaly pipeline.",
                extra={
                    "datetime_column": dt_cols[0],
                    "value_columns": value_cols[:5],
                    "suggested_targets": value_cols[:3],
                },
            )

        # Weak text signal fallback
        if text_cols:
            return self._result(
                "text",
                "text_classification",
                "Text column detected — NLP classification pipeline.",
                extra={"text_column": text_cols[0], "suggested_targets": self._label_columns(df, exclude=text_cols)},
            )

        return self._result(
            "tabular",
            "tabular_automl",
            "Standard tabular CSV/Excel — classification/regression AutoML.",
            extra={"suggested_targets": self._label_columns(df)},
        )

    def _text_columns(self, df: pd.DataFrame, cols: list) -> list[str]:
        """Rank likely free-text columns; exclude ids (e.g. tweet_id)."""
        id_pattern = re.compile(r"(^id$|_id$|\bid\b|tweet_id|user_id)", re.I)
        ranked: list[tuple[float, str]] = []
        for c in cols:
            name = str(c).lower()
            if id_pattern.search(name.replace(" ", "_")):
                continue
            if not (TEXT_COL_PATTERNS.search(name) or self._is_text_column(df[c])):
                continue
            avg_len = float(df[c].astype(str).str.len().mean())
            score = avg_len
            if name == "text" or name.endswith("_text"):
                score += 1000
            elif name in ("message", "body", "content", "review", "comment"):
                score += 500
            ranked.append((score, str(c)))
        ranked.sort(key=lambda x: -x[0])
        return [c for _, c in ranked]

    @staticmethod
    def _datetime_columns(df: pd.DataFrame, cols: list) -> list[str]:
        """Columns that look like real timestamps (not timezone/location metadata)."""
        ts_name = re.compile(r"(timestamp|datetime|(^date$|_date$|^date_))", re.I)
        found: list[str] = []
        for c in cols:
            name = str(c).lower()
            if TIME_COL_FALSE_POSITIVE.search(name):
                continue
            name_match = TIME_COL_PATTERNS.search(name) or pd.api.types.is_datetime64_any_dtype(df[c])
            if not name_match:
                continue
            series = df[c]
            if pd.api.types.is_numeric_dtype(series):
                # Integers like tenure_months (38) parse as 1970-era junk — require explicit ts name or epoch-scale values
                if not ts_name.search(name):
                    continue
                vmax = float(series.max(skipna=True)) if series.notna().any() else 0.0
                if vmax < 1e8:
                    continue
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            valid_ratio = float(parsed.notna().mean())
            if valid_ratio < 0.5:
                continue
            years = parsed.dt.year.dropna()
            if years.empty:
                continue
            # Reject integer-as-nanoseconds false positives (everything lands in 1970)
            if years.median() < 1980 or years.max() > 2100:
                continue
            found.append(str(c))
        return found

    @staticmethod
    def _timeseries_value_columns(df: pd.DataFrame, cols: list) -> list[str]:
        """Numeric columns suitable for forecasting (exclude ids/confidence scores)."""
        skip_name = re.compile(r"(id|confidence|index|count|gold|code|zip|lat|lon)", re.I)
        values: list[str] = []
        for c in df.select_dtypes(include=["number"]).columns:
            name = str(c).lower()
            if skip_name.search(name):
                continue
            if df[c].nunique(dropna=True) < 5:
                continue
            values.append(str(c))
        return values

    @staticmethod
    def _looks_like_time_index(df: pd.DataFrame, dt_col: str) -> bool:
        """True when rows look like sequential observations indexed by time (not one row per entity)."""
        id_pattern = re.compile(r"(^id$|_id$|\bid\b|customer|user|account|order|transaction)", re.I)
        has_entity_id = any(id_pattern.search(str(c).replace(" ", "_")) for c in df.columns if str(c) != dt_col)
        if has_entity_id:
            return False
        parsed = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        n_valid = int(parsed.notna().sum())
        if n_valid < 10:
            return False
        uniq_ratio = parsed.nunique(dropna=True) / max(n_valid, 1)
        return uniq_ratio >= 0.5

    @staticmethod
    def _is_text_column(series: pd.Series) -> bool:
        if series.dtype not in ("object", "string"):
            return False
        sample = series.dropna().astype(str).head(50)
        if sample.empty:
            return False
        return sample.str.len().mean() > 35 and sample.nunique() > 5

    @staticmethod
    def _label_columns(df: pd.DataFrame, exclude: Optional[list[str]] = None) -> list[str]:
        exclude = set(exclude or [])
        candidates = []
        for c in df.columns:
            if c in exclude:
                continue
            n = df[c].nunique(dropna=True)
            if 2 <= n <= 50:
                candidates.append(str(c))
        return candidates[:10]

    @staticmethod
    def _pick_text_column(df: pd.DataFrame) -> Optional[str]:
        for c in df.columns:
            if LOG_COL_PATTERNS.search(str(c).lower()):
                if df[c].dtype == object or str(df[c].dtype) == "string":
                    return str(c)
        obj = df.select_dtypes(include=["object"]).columns
        return str(obj[0]) if len(obj) else None

    @staticmethod
    def _result(modality: str, pipeline: str, reason: str, extra: Optional[dict] = None) -> dict[str, Any]:
        base = {
            "modality": modality,
            "pipeline_type": pipeline,
            "detection_reason": reason,
            "preprocessing": _PREPROCESSING.get(modality, []),
            "models": _MODELS.get(modality, []),
            "metrics": _METRICS.get(modality, []),
            "explainability": _EXPLAIN.get(modality, []),
            "drift": _DRIFT.get(modality, []),
        }
        if extra:
            base.update(extra)
        return base


_PREPROCESSING = {
    "tabular": ["missing imputation", "outlier capping", "encoding", "feature engineering", "feature selection"],
    "text": ["lowercasing", "punctuation removal", "tokenization", "TF-IDF embeddings", "n-gram features"],
    "image": ["resize", "normalize", "augmentation (optional)", "CNN embeddings or histogram features"],
    "timeseries": ["datetime parsing", "resampling", "lag features", "rolling statistics", "seasonal decomposition"],
    "logs": ["timestamp parsing", "log parsing", "tokenization", "TF-IDF", "severity normalization"],
    "documents": ["PDF text extraction", "table extraction", "chunking", "TF-IDF / embeddings for RAG"],
}

_MODELS = {
    "tabular": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Optuna tuning"],
    "text": ["TF-IDF + Logistic", "TF-IDF + LightGBM", "TF-IDF + XGBoost", "Linear SVM", "Multinomial NB"],
    "image": ["PCA + LightGBM", "PCA + XGBoost", "PCA + HistGBM", "PCA + Random Forest", "PCA + Logistic"],
    "timeseries": ["Lag + LightGBM", "Lag + XGBoost", "Lag + HistGBM", "Lag + Random Forest", "Lag + Ridge"],
    "logs": ["TF-IDF + LightGBM/XGBoost", "TF-IDF + classifier", "KMeans incident clustering"],
    "documents": ["TF-IDF + LightGBM/XGBoost", "document classifier", "RAG retriever"],
}

_METRICS = {
    "tabular": ["accuracy", "F1", "precision", "recall", "ROC-AUC", "R²", "RMSE", "MAE"],
    "text": ["accuracy", "F1", "precision", "recall", "confusion matrix"],
    "image": ["accuracy", "F1", "top-k accuracy", "confusion matrix"],
    "timeseries": ["MAE", "RMSE", "MAPE", "sMAPE", "R²"],
    "logs": ["accuracy", "F1", "silhouette (clustering)", "MTTR proxy metrics"],
    "documents": ["classification accuracy", "retrieval precision@k", "ROUGE (summarization)"],
}

_EXPLAIN = {
    "tabular": ["SHAP global/local", "feature importance", "partial dependence"],
    "text": ["top keywords per class", "TF-IDF weights", "SHAP on sparse features", "attention highlights (deep)"],
    "image": ["Grad-CAM", "saliency maps", "SHAP on embeddings"],
    "timeseries": ["lag importance", "SHAP on engineered features", "forecast residual analysis"],
    "logs": ["important tokens", "cluster themes", "RCA keyword groups"],
    "documents": ["highlighted passages", "retrieved chunks", "citation-style evidence"],
}

_DRIFT = {
    "tabular": ["Evidently column drift", "PSI", "feature distribution shift"],
    "text": ["vocabulary drift", "embedding drift", "term frequency shift"],
    "image": ["pixel histogram drift", "embedding centroid shift", "concept drift"],
    "timeseries": ["distribution shift", "seasonality change", "anomaly rate drift"],
    "logs": ["new error templates", "volume spike", "assignment pattern drift"],
    "documents": ["topic drift", "vocabulary shift", "retrieval quality decay"],
}


def save_modality_metadata(upload_dir: Path, dataset_id: str, meta: dict[str, Any]) -> Path:
    path = upload_dir / f"{dataset_id}.meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_modality_metadata(upload_dir: Path, dataset_id: str) -> dict[str, Any]:
    path = upload_dir / f"{dataset_id}.meta.json"
    if not path.exists():
        return {"modality": "tabular", "pipeline_type": "tabular_automl"}
    return json.loads(path.read_text(encoding="utf-8"))
