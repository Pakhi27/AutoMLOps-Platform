"""Text modality pipeline — TF-IDF + sklearn classifiers."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from app.core.config import get_settings
from app.services.modality.base import BaseModalityPipeline
from app.utils.io_utils import read_csv_safely


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class TextModalityPipeline(BaseModalityPipeline):
    modality = "text"
    pipeline_type = "text_classification"

    def run(
        self,
        job_id: str,
        dataset_path: str,
        target_column: str,
        metadata: dict[str, Any],
        text_column: Optional[str] = None,
        test_size: Optional[float] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        t0 = time.time()
        settings = get_settings()
        test_size = test_size if test_size is not None else settings.modeling.get("test_size", 0.2)
        df = read_csv_safely(Path(dataset_path))
        text_col = text_column or metadata.get("text_column")
        if not text_col:
            for c in df.columns:
                if c != target_column and df[c].dtype == object:
                    if df[c].astype(str).str.len().mean() > 30:
                        text_col = c
                        break
        if not text_col or text_col not in df.columns:
            raise ValueError("Text column not found. Set text_column in request or metadata.")

        self.report("preprocessing", "Cleaning and tokenizing text", 15)
        df = df.dropna(subset=[text_col, target_column])
        X_text = df[text_col].astype(str).map(_clean_text)
        y_raw = df[target_column].astype(str)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)

        from app.utils.split_utils import stratify_vector, validate_classification_target

        validate_classification_target(y)
        stratify = stratify_vector(y, test_size=test_size)
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y, test_size=test_size, random_state=42, stratify=stratify
        )
        y_test_labels = label_encoder.inverse_transform(y_test)

        self.report("preprocessing", "Building TF-IDF embeddings", 30)
        text_cfg = settings.modalities.get("text", {})
        max_features = text_cfg.get("max_tfidf_features", 8000)
        ngram_range = tuple(text_cfg.get("ngram_range", [1, 2]))

        def _tfidf_clf(clf) -> Pipeline:
            return Pipeline([
                ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)),
                ("clf", clf),
            ])

        candidates = {
            "tfidf_logistic": _tfidf_clf(LogisticRegression(max_iter=1000, class_weight="balanced")),
            "tfidf_lgbm": _tfidf_clf(LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=-1, class_weight="balanced", verbose=-1,
            )),
            "tfidf_xgboost": _tfidf_clf(XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6, eval_metric="mlogloss",
                tree_method="hist", n_jobs=-1,
            )),
            "tfidf_svc": _tfidf_clf(LinearSVC(class_weight="balanced")),
            "tfidf_nb": _tfidf_clf(MultinomialNB()),
        }

        self.report("training", "Training text classifiers", 55)
        best_name, best_model, best_f1 = "", None, -1.0
        leaderboard: dict[str, float] = {}
        for name, pipe in candidates.items():
            try:
                pipe.fit(X_train, y_train)
                pred = label_encoder.inverse_transform(pipe.predict(X_test).astype(int))
            except Exception:
                continue
            f1 = float(f1_score(y_test_labels, pred, average="weighted", zero_division=0))
            leaderboard[name] = round(f1, 4)
            if f1 > best_f1:
                best_f1, best_name, best_model = f1, name, pipe

        if best_model is None:
            raise ValueError("All text classifiers failed to train. Check labels and text column.")

        pred = label_encoder.inverse_transform(best_model.predict(X_test).astype(int))
        metrics = {
            "accuracy": round(float(accuracy_score(y_test_labels, pred)), 4),
            "f1_weighted": round(float(f1_score(y_test_labels, pred, average="weighted", zero_division=0)), 4),
            "precision_weighted": round(float(precision_score(y_test_labels, pred, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_test_labels, pred, average="weighted", zero_division=0)), 4),
        }

        self.report("explainability", "Extracting top keywords per class", 80)
        explainability = self._keyword_explain(best_model, best_name, label_encoder)

        out_dir = settings.artifacts_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "text_pipeline.joblib"
        joblib.dump({
            "model": best_model,
            "text_column": text_col,
            "target_column": target_column,
            "label_classes": label_encoder.classes_.tolist(),
        }, model_path)
        (out_dir / "text_explainability.json").write_text(json.dumps(explainability, indent=2), encoding="utf-8")

        self.report("complete", "Text pipeline complete", 100)
        return {
            "job_id": job_id,
            "modality": self.modality,
            "pipeline_type": self.pipeline_type,
            "task_type": "classification",
            "model_name": best_name,
            "text_column": text_col,
            "target_column": target_column,
            "label_classes": label_encoder.classes_.tolist(),
            "metrics": metrics,
            "baseline_scores": leaderboard,
            "explainability": explainability,
            "drift_notes": ["Monitor vocabulary drift", "TF-IDF term frequency shift", "embedding centroid drift"],
            "pipeline_path": str(model_path),
            "training_seconds": round(time.time() - t0, 1),
        }

    @staticmethod
    def _keyword_explain(model: Pipeline, model_name: str, label_encoder: LabelEncoder | None = None) -> dict[str, Any]:
        try:
            tfidf: TfidfVectorizer = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            terms = np.array(tfidf.get_feature_names_out())
            if hasattr(clf, "coef_"):
                coef = clf.coef_
                if coef.ndim == 1:
                    coef = coef.reshape(1, -1)
                top_per_class = {}
                classes = list(label_encoder.classes_) if label_encoder is not None else []
                for i in range(coef.shape[0]):
                    idx = np.argsort(coef[i])[-15:][::-1]
                    key = classes[i] if i < len(classes) else f"class_{i}"
                    top_per_class[key] = terms[idx].tolist()
                return {"method": "tfidf_coefficients", "top_keywords": top_per_class, "model": model_name}
        except Exception:
            pass
        return {"method": "tfidf", "note": "Keyword explainability available after training", "model": model_name}
