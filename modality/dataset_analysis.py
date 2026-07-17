"""Resolve uploaded dataset files and build profiles for non-tabular data."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.services.modality.detector import load_modality_metadata
from app.utils.io_utils import read_csv_safely


def resolve_dataset_path(upload_dir: Path, dataset_id: str) -> Optional[Path]:
    """Find the on-disk upload for a dataset id (csv, zip, txt, pdf, etc.)."""
    meta = load_modality_metadata(upload_dir, dataset_id)
    modality = meta.get("modality", "tabular")

    by_modality: dict[str, list[str]] = {
        "image": [".zip", ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"],
        "documents": [".pdf", ".txt", ".md"],
        "text": [".csv", ".tsv", ".jsonl", ".txt"],
        "tabular": [".csv", ".tsv", ".xlsx", ".xls"],
        "timeseries": [".csv", ".tsv", ".xlsx", ".xls"],
        "logs": [".csv", ".tsv", ".jsonl"],
    }
    preferred = by_modality.get(modality, [])
    for ext in preferred:
        p = upload_dir / f"{dataset_id}{ext}"
        if p.exists():
            return p

    patterns = [
        f"{dataset_id}.csv",
        f"{dataset_id}.zip",
        f"{dataset_id}.xlsx",
        f"{dataset_id}.xls",
        f"{dataset_id}.tsv",
        f"{dataset_id}.pdf",
        f"{dataset_id}.txt",
        f"{dataset_id}.jsonl",
        f"{dataset_id}.jpg",
        f"{dataset_id}.jpeg",
        f"{dataset_id}.png",
    ]
    for name in patterns:
        p = upload_dir / name
        if p.exists():
            return p
    matches = sorted(upload_dir.glob(f"{dataset_id}.*"), key=lambda x: x.suffix)
    return matches[0] if matches else None


def load_tabular_dataframe(upload_dir: Path, dataset_id: str) -> Optional[pd.DataFrame]:
    path = resolve_dataset_path(upload_dir, dataset_id)
    if path is None:
        return None
    ext = path.suffix.lower()
    if ext == ".csv":
        return read_csv_safely(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return None


def _count_images_in_zip(zip_path: Path) -> dict[str, Any]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    class_counts: dict[str, int] = {}
    total = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            p = Path(name)
            if p.suffix.lower() not in image_exts:
                continue
            parts = [x for x in p.parts if x and x != "/"]
            label = parts[-2] if len(parts) >= 2 else "unknown"
            class_counts[label] = class_counts.get(label, 0) + 1
            total += 1
    return {"n_images": total, "class_counts": class_counts, "n_classes": len(class_counts)}


def _count_images_in_folder(root: Path) -> dict[str, Any]:
    from app.services.modality.image_pipeline import _collect_images

    pairs = _collect_images(root)
    class_counts: dict[str, int] = {}
    for _, label in pairs:
        class_counts[label] = class_counts.get(label, 0) + 1
    return {"n_images": len(pairs), "class_counts": class_counts, "n_classes": len(class_counts)}


def summarize_image_dataset(upload_dir: Path, dataset_id: str) -> dict[str, Any]:
    path = resolve_dataset_path(upload_dir, dataset_id)
    if path is None:
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found.")
    extract_dir = upload_dir / f"{dataset_id}_images"
    if path.suffix.lower() == ".zip":
        if extract_dir.exists():
            summary = _count_images_in_folder(extract_dir)
        else:
            summary = _count_images_in_zip(path)
    elif extract_dir.exists():
        summary = _count_images_in_folder(extract_dir)
    else:
        summary = _count_images_in_folder(path.parent if path.is_file() else path)
    return summary


def build_image_profile(
    upload_dir: Path,
    dataset_id: str,
    target_column: str = "label",
    meta: Optional[dict] = None,
) -> dict[str, Any]:
    meta = meta or load_modality_metadata(upload_dir, dataset_id)
    summary = summarize_image_dataset(upload_dir, dataset_id)
    counts = summary["class_counts"]
    total = summary["n_images"]
    max_c = max(counts.values()) if counts else 0
    min_c = min(counts.values()) if counts else 0
    imbalance_ratio = max_c / total if total else 1.0
    return {
        "modality": "image",
        "pipeline_type": meta.get("pipeline_type", "image_classification"),
        "n_rows": total,
        "n_columns": 2,
        "numeric_columns": [],
        "categorical_columns": ["label"],
        "datetime_columns": [],
        "missing_values": {},
        "n_duplicate_rows": 0,
        "class_distribution": counts,
        "target_analysis": {
            "type": "classification",
            "n_classes": summary["n_classes"],
            "is_imbalanced": imbalance_ratio > 0.7 if total else False,
            "imbalance_ratio": round(imbalance_ratio, 4),
            "class_counts": counts,
        },
        "correlation_with_target": {},
        "detection_reason": meta.get("detection_reason", "Image folder dataset"),
        "target_column": target_column,
    }


def score_image_quality(summary: dict[str, Any]) -> dict[str, Any]:
    total = summary["n_images"]
    counts = summary["class_counts"]
    n_classes = summary["n_classes"]

    def grade(score: float) -> str:
        if score >= 90:
            return "A"
        if score >= 75:
            return "B"
        if score >= 60:
            return "C"
        if score >= 45:
            return "D"
        return "F"

    size_score = min(100, 30 + total * 2)
    balance_score = 95
    if counts and total:
        max_share = max(counts.values()) / total
        balance_score = max(30, 100 - max_share * 80)
    class_score = min(100, n_classes * 25) if n_classes >= 2 else 40

    dimensions = {
        "sample_size": {
            "score": round(size_score, 1),
            "grade": grade(size_score),
            "detail": f"{total} images across {n_classes} classes",
        },
        "class_balance": {
            "score": round(balance_score, 1),
            "grade": grade(balance_score),
            "detail": str(counts),
        },
        "class_coverage": {
            "score": round(class_score, 1),
            "grade": grade(class_score),
            "detail": f"{n_classes} label folders detected",
        },
        "leakage": {
            "score": 100.0,
            "grade": "A",
            "detail": "Folder-based labels — no tabular leakage scan (use holdout split)",
        },
        "missing_values": {
            "score": 100.0,
            "grade": "A",
            "detail": "N/A for image archives",
        },
    }
    overall = round(sum(d["score"] for d in dimensions.values()) / len(dimensions), 1)
    return {
        "overall_score": overall,
        "overall_grade": grade(overall),
        "dimensions": dimensions,
        "suggestions": [
            "Use at least 50+ images per class for stronger CNN/transfer learning",
            "Keep class folders balanced (similar image counts)",
            "Zip structure: class_name/image.jpg",
        ],
    }


def image_leakage_report(summary: dict[str, Any], target_column: str) -> dict[str, Any]:
    return {
        "leakage_detected": False,
        "n_issues": 0,
        "issues": [],
        "recommended_drop": [],
        "summary": (
            f"Image dataset ({summary['n_images']} images, {summary['n_classes']} classes). "
            "Tabular leakage checks do not apply — validate train/test splits and duplicate images across folders."
        ),
        "target_column": target_column,
        "modality": "image",
    }
