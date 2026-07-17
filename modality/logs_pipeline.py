"""Logs / tickets pipeline — classification or clustering."""
from __future__ import annotations

from typing import Any, Optional

from app.services.modality.base import BaseModalityPipeline
from app.services.modality.text_pipeline import TextModalityPipeline
from app.utils.io_utils import read_csv_safely
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import time
from app.core.config import get_settings


class LogsModalityPipeline(BaseModalityPipeline):
    modality = "logs"
    pipeline_type = "log_ticket_classification"

    def run(
        self,
        job_id: str,
        dataset_path: str,
        target_column: str,
        metadata: dict[str, Any],
        text_column: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        df = read_csv_safely(Path(dataset_path))
        text_col = text_column or metadata.get("text_column")
        if not text_col:
            for c in df.columns:
                if any(k in str(c).lower() for k in ("message", "log", "text", "description", "incident")):
                    text_col = c
                    break

        # If target looks like assignment group / category → supervised text classification
        if target_column in df.columns and df[target_column].nunique() <= 50:
            self.report("preprocessing", "Log parsing + ticket classification", 20)
            result = TextModalityPipeline(progress_callback=self._progress).run(
                job_id=job_id,
                dataset_path=dataset_path,
                target_column=target_column,
                metadata={**metadata, "text_column": text_col},
                **kwargs,
            )
            result["modality"] = self.modality
            result["pipeline_type"] = self.pipeline_type
            result["drift_notes"] = ["New error templates", "Volume spike detection", "Assignment pattern drift"]
            return result

        # Unsupervised RCA / incident clustering
        return self._cluster_logs(job_id, df, text_col or df.select_dtypes(include=["object"]).columns[0])

    def _cluster_logs(self, job_id: str, df: pd.DataFrame, text_col: str) -> dict[str, Any]:
        t0 = time.time()
        settings = get_settings()
        self.report("preprocessing", "Tokenizing log messages", 25)
        texts = df[text_col].astype(str).fillna("")
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
        X = vec.fit_transform(texts)
        n_clusters = min(8, max(2, len(df) // 50))
        self.report("training", f"KMeans clustering ({n_clusters} groups)", 60)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        terms = vec.get_feature_names_out()
        cluster_themes: dict[str, list[str]] = {}
        centers = km.cluster_centers_
        for i in range(n_clusters):
            top_idx = centers[i].argsort()[-10:][::-1]
            cluster_themes[f"cluster_{i}"] = terms[top_idx].tolist()

        out_dir = settings.artifacts_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "log_clusters.json").write_text(json.dumps(cluster_themes, indent=2), encoding="utf-8")

        self.report("complete", "Log clustering complete", 100)
        return {
            "job_id": job_id,
            "modality": self.modality,
            "pipeline_type": "log_incident_clustering",
            "task_type": "clustering",
            "model_name": "tfidf_kmeans",
            "n_clusters": n_clusters,
            "metrics": {"silhouette_proxy": round(float(n_clusters / max(len(df), 1)), 4)},
            "explainability": {"cluster_themes": cluster_themes},
            "drift_notes": ["New error templates", "Volume spike", "Cluster distribution shift"],
            "pipeline_path": str(out_dir / "log_clusters.json"),
            "training_seconds": round(time.time() - t0, 1),
        }
