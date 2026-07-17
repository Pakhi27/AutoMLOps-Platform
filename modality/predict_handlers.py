"""Modality-aware prediction helpers for non-tabular models."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd

from app.services.modality.image_utils import is_image_file, load_image_vector, resolve_image_path


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _resolve_label_classes(bundle: dict, entry: dict, clf: Any = None) -> list[str] | None:
    classes = bundle.get("label_classes") or entry.get("label_classes")
    if classes:
        return [str(c) for c in classes]
    if clf is not None and hasattr(clf, "classes_"):
        raw = list(clf.classes_)
        if raw and not isinstance(raw[0], (int, np.integer)):
            return [str(c) for c in raw]
    return None


def _decode_class_predictions(raw_preds: Any, label_classes: list[str] | None) -> list[str]:
    decoded: list[str] = []
    for p in raw_preds:
        try:
            idx = int(p)
            if label_classes and 0 <= idx < len(label_classes):
                decoded.append(label_classes[idx])
            else:
                decoded.append(str(p))
        except (TypeError, ValueError):
            decoded.append(str(p))
    return decoded


def _format_class_probabilities(
    proba_rows: list[list[float]] | None,
    label_classes: list[str] | None,
) -> list[dict[str, float]] | None:
    if not proba_rows:
        return None
    formatted: list[dict[str, float]] = []
    for row in proba_rows:
        if label_classes and len(label_classes) == len(row):
            formatted.append({label_classes[i]: round(float(row[i]), 4) for i in range(len(row))})
        else:
            formatted.append({str(i): round(float(v), 4) for i, v in enumerate(row)})
    return formatted


def _extract_texts(records: list[dict], text_column: str | None) -> list[str]:
    texts: list[str] = []
    for row in records:
        if text_column and text_column in row:
            texts.append(_clean_text(row[text_column]))
            continue
        # Common aliases
        for key in ("review_text", "text", "message", "content", "body", "review"):
            if key in row:
                texts.append(_clean_text(row[key]))
                break
        else:
            # First non-empty string field
            val = next((v for v in row.values() if isinstance(v, str) and v.strip()), "")
            texts.append(_clean_text(val))
    return texts


def predict_modality(entry: dict, artifact: Any, df: pd.DataFrame, explain: bool = False) -> dict:
    modality = entry.get("modality", "tabular")

    if modality == "text" or (isinstance(artifact, dict) and "text_column" in artifact):
        bundle = artifact if isinstance(artifact, dict) else {}
        model = bundle.get("model", artifact)
        text_col = bundle.get("text_column") or entry.get("text_column")
        texts = _extract_texts(df.to_dict(orient="records"), text_col)
        raw_preds = model.predict(texts)
        label_classes = _resolve_label_classes(
            bundle,
            entry,
            model.named_steps.get("clf") if hasattr(model, "named_steps") else model,
        )
        predictions = _decode_class_predictions(raw_preds, label_classes)
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = _format_class_probabilities(
                    model.predict_proba(texts).tolist(),
                    label_classes,
                )
            except Exception:
                pass
        explanations = None
        if explain:
            explanations = _text_explanations(entry, bundle, predictions[0] if predictions else "")
        return {"predictions": predictions, "probabilities": probabilities, "explanations": explanations}

    if modality == "timeseries" or (isinstance(artifact, dict) and "feature_columns" in artifact and "datetime_column" in artifact):
        bundle = artifact
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
        # User must pass engineered features OR raw value for simple single-step
        if all(c in df.columns for c in feature_columns):
            X = df[feature_columns]
        else:
            raise ValueError(
                f"Time-series predict needs lag features: {feature_columns[:5]}... "
                "Pass engineered columns or use batch CSV with features."
            )
        preds = model.predict(X)
        return {"predictions": [float(p) for p in preds], "probabilities": None, "explanations": None}

    if modality == "image" or (isinstance(artifact, dict) and "pca" in artifact and "classifier" in artifact):
        bundle = artifact
        pca = bundle["pca"]
        clf = bundle["classifier"]
        size = tuple(bundle.get("image_size", [64, 64]))
        rgb = bool(bundle.get("rgb", True))
        paths: list[Path] = []
        for row in df.to_dict(orient="records"):
            for key in ("image_path", "path", "file", "filepath", "image"):
                if key in row and row[key]:
                    paths.append(resolve_image_path(str(row[key])))
                    break
            else:
                val = next((v for v in row.values() if isinstance(v, str) and str(v).strip()), "")
                if val:
                    paths.append(resolve_image_path(str(val)))
        if not paths:
            raise ValueError("Provide image_path with a valid local image file path.")
        vectors = []
        for p in paths:
            vec = load_image_vector(p, size=size, rgb=rgb)
            if vec is None:
                raise ValueError(f"Could not load image: {p}")
            vectors.append(vec)
        X_pca = pca.transform(np.vstack(vectors))
        raw_preds = clf.predict(X_pca)
        label_classes = _resolve_label_classes(bundle, entry, clf)
        predictions = _decode_class_predictions(raw_preds, label_classes)
        probabilities = None
        if hasattr(clf, "predict_proba"):
            try:
                probabilities = _format_class_probabilities(
                    clf.predict_proba(X_pca).tolist(),
                    label_classes,
                )
            except Exception:
                pass
        image_rows = []
        for i, path in enumerate(paths):
            image_rows.append({
                "image_path": str(path),
                "preview_url": f"/predict/preview-image?path={quote(str(path), safe='')}",
                "prediction": predictions[i] if i < len(predictions) else None,
                "probabilities": probabilities[i] if probabilities and i < len(probabilities) else None,
            })
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "explanations": None,
            "label_classes": label_classes,
            "image_rows": image_rows,
        }

    if modality == "image" or (isinstance(artifact, dict) and "pca" in artifact):
        raise ValueError("Image artifact missing classifier. Retrain the image model.")

    raise ValueError(f"Unsupported modality for predict: {modality}")


def load_artifact(pipeline_path: str) -> Any:
    return joblib.load(pipeline_path)


def _text_explanations(entry: dict, bundle: dict, prediction: str) -> list[dict[str, Any]]:
    """Return top keywords for predicted class from saved text explainability artifact."""
    job_id = entry.get("job_id")
    if not job_id:
        return [{"note": "Text model — keyword explainability from training"}]
    from app.core.config import get_settings

    path = get_settings().artifacts_dir / job_id / "text_explainability.json"
    if not path.exists():
        return [{"note": "Keyword explainability not found — retrain to generate"}]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        keywords = data.get("top_keywords", {})
        if not keywords:
            return [{"note": data.get("note", "No keywords available")}]

        class_idx = None
        model = bundle.get("model")
        if model is not None and hasattr(model, "classes_"):
            classes = list(model.classes_)
            try:
                class_idx = classes.index(prediction)
            except ValueError:
                for i, c in enumerate(classes):
                    if str(c).lower() == str(prediction).lower():
                        class_idx = i
                        break
        if class_idx is None and entry.get("label_classes"):
            try:
                class_idx = entry["label_classes"].index(prediction)
            except ValueError:
                pass

        if class_idx is not None:
            words = keywords.get(f"class_{class_idx}")
            if words:
                return [{w: 1.0 for w in words[:10]}]

        first = next(iter(keywords.values()), [])
        return [{w: 1.0 for w in first[:10]}] if first else [{"note": "No keywords found"}]
    except Exception:
        return [{"note": "Could not load text explainability"}]
