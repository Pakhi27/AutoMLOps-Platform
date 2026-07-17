"""Image classification pipeline — RGB features + strong sklearn / boosting models."""
from __future__ import annotations

import json
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from app.core.config import get_settings
from app.services.modality.base import BaseModalityPipeline
from app.services.modality.image_utils import load_image_vector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
GENERIC_FOLDER_NAMES = {
    "images", "image", "train", "test", "val", "valid", "data", "imgs", "photos",
    "seg_train", "seg_test", "training", "dataset", "my_images", "training_set",
    "petimages", "pets", "cats_and_dogs", "animals",
}


def _dir_has_images(d: Path) -> bool:
    return any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in d.iterdir())


def _collect_from_class_folders(root: Path) -> list[tuple[str, str]]:
    """Return (path, label) using immediate subfolder names as labels."""
    pairs: list[tuple[str, str]] = []
    if not root.is_dir():
        return pairs
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = _label_from_folder_name(class_dir.name)
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                pairs.append((str(f), label))
    return pairs


def _collect_from_class_folders_nested(root: Path) -> list[tuple[str, str]]:
    """Support dog/cat, train/dog/cat, and my_images/dog/cat ZIP layouts."""
    pairs: list[tuple[str, str]] = []
    if not root.is_dir():
        return pairs

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        label = _label_from_folder_name(entry.name)
        direct_images = [f for f in entry.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if direct_images:
            for f in sorted(direct_images):
                pairs.append((str(f), label))
            continue

        subdirs = [s for s in entry.iterdir() if s.is_dir() and _dir_has_images(s)]
        if not subdirs:
            continue
        parent_key = entry.name.lower().replace(" ", "_")
        if parent_key in GENERIC_FOLDER_NAMES or len(subdirs) >= 2:
            for sub in sorted(subdirs):
                sub_label = _label_from_folder_name(sub.name)
                for f in sorted(sub.iterdir()):
                    if f.suffix.lower() in IMAGE_EXTS:
                        pairs.append((str(f), sub_label))
    return pairs


def _best_class_folder_pairs(root: Path) -> list[tuple[str, str]]:
    """Scan root and nested dirs; pick the layout with the most valid class labels."""
    from collections import Counter

    candidates: list[list[tuple[str, str]]] = []
    seen: set[str] = set()
    for base in [root, *sorted(root.rglob("*"))]:
        if not base.is_dir() or str(base) in seen:
            continue
        seen.add(str(base))
        for collector in (_collect_from_class_folders_nested, _collect_from_class_folders):
            pairs = collector(base)
            labels = {lbl for _, lbl in pairs}
            if len(labels) >= 2:
                candidates.append(pairs)

    if not candidates:
        return []

    def score(p: list[tuple[str, str]]) -> tuple:
        labels = {lbl for _, lbl in p}
        counts = Counter(lbl for _, lbl in p)
        min_c = min(counts.values())
        generic = sum(1 for lbl in labels if lbl in GENERIC_FOLDER_NAMES)
        numeric = sum(1 for lbl in labels if str(lbl).isdigit())
        return (min_c >= 2, len(p), -numeric, -generic, len(labels))

    return max(candidates, key=score)


def _is_numeric_label(name: str) -> bool:
    return bool(re.fullmatch(r"\d+", str(name).strip()))


def _label_from_folder_name(name: str) -> str:
    """e.g. n02099601-shiba_inu -> shiba_inu, Dog -> dog."""
    cleaned = name.strip()
    synset = re.match(r"^n\d+-(.+)$", cleaned, re.I)
    if synset:
        return synset.group(1).lower().replace(" ", "_")
    return cleaned.lower().replace(" ", "_")


def _label_from_filename(stem: str) -> str:
    """Derive class from filename when folder labels are generic."""
    lower = stem.lower()
    # dogs vs cats: dog.102.jpg, cat.0.jpg
    binary = re.match(r"^(dog|cat)\.(\d+)$", lower)
    if binary:
        return binary.group(1)
    # breed_123: shiba_inu_29, golden_retriever_105
    numbered = re.match(r"^(.+)_(\d+)$", lower)
    if numbered:
        prefix = numbered.group(1)
        if prefix not in GENERIC_FOLDER_NAMES:
            return prefix.replace(" ", "_")
    return lower.replace(" ", "_")


def _walk_image_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
        return [root]
    files: list[Path] = []
    skip_parts = {"__macosx", ".git", ".svn"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        parts_lower = {p.lower() for p in path.parts}
        if parts_lower & skip_parts or path.name.startswith("."):
            continue
        files.append(path)
    return files


def _label_for_image_path(image_path: Path, scan_root: Path) -> str:
    """Derive class from the nearest non-generic folder in the path, else filename."""
    try:
        folder_parts = list(image_path.relative_to(scan_root).parts[:-1])
    except ValueError:
        folder_parts = []
    for part in reversed(folder_parts):
        key = _label_from_folder_name(part)
        if key in GENERIC_FOLDER_NAMES:
            continue
        if _is_numeric_label(key):
            # Numbered class folder (0/, 219/) — keep label so validation can reject bad layouts
            if part == folder_parts[-1]:
                return key
            continue
        return key
    return _label_from_filename(image_path.stem)


def _collect_by_path_labels(root: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for f in _walk_image_files(root):
        pairs.append((str(f), _label_for_image_path(f, root)))
    return pairs


def _good_label_set(pairs: list[tuple[str, str]], min_images: int = 10) -> bool:
    from collections import Counter

    if len(pairs) < min_images:
        return False
    counts = Counter(label for _, label in pairs)
    if len(counts) < 2:
        return False
    n_classes = len(counts)
    n_images = len(pairs)
    if n_classes / n_images > 0.5 and max(counts.values()) == 1:
        return False
    numeric = sum(1 for lbl in counts if _is_numeric_label(str(lbl)))
    if numeric / n_classes > 0.7 and max(counts.values()) == 1:
        return False
    return True


def _resolve_scan_root(extract_dir: Path) -> Path:
    """Unwrap single top-level folder from ZIP extract (common in Kaggle downloads)."""
    root = extract_dir
    for _ in range(4):
        subdirs = [
            p for p in root.iterdir()
            if p.is_dir() and p.name.lower() not in {"__macosx", ".git"} and not p.name.startswith(".")
        ]
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if files or len(subdirs) != 1:
            break
        root = subdirs[0]
    return root


def _collect_from_filename_labels(root: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for f in _walk_image_files(root):
        pairs.append((str(f), _label_from_filename(f.stem)))
    return pairs


def _validate_image_labels(pairs: list[tuple[str, str]]) -> None:
    """Ensure folder/filename layout yields real classes, not one label per image."""
    from collections import Counter

    counts = Counter(label for _, label in pairs)
    n_images = len(pairs)
    n_classes = len(counts)
    min_count = min(counts.values())
    numeric_label_ratio = sum(1 for lbl in counts if str(lbl).isdigit()) / max(n_classes, 1)

    if n_classes < 2:
        raise ValueError(
            "Need at least 2 image classes. Use separate folders per class (dog/, cat/) "
            "or breed-style filenames (shiba_inu_01.jpg)."
        )

    # Numbered folders/files → one class per image (0/, 219/, 219.jpg)
    if n_classes / n_images > 0.5 or (numeric_label_ratio > 0.7 and min_count == 1):
        raise ValueError(
            f"Image ZIP layout issue: {n_classes} classes from {n_images} images — "
            "almost every image has its own label (numbered folders like 0/, 219/ or files like 219.jpg). "
            "Your ZIP may use train/dog/ and train/cat/ — re-upload after restarting the server. "
            "Or flatten to:\n"
            "  dog/…  cat/…\n"
            "Each class needs at least 2 images."
        )

    if min_count < 2:
        sparse = [lbl for lbl, n in counts.items() if n < 2]
        preview = ", ".join(str(x) for x in sparse[:6])
        extra = f" (+{len(sparse) - 6} more)" if len(sparse) > 6 else ""
        raise ValueError(
            f"Some image classes have only 1 photo (e.g. {preview}{extra}). "
            "Add more images per class folder or group filenames by breed prefix."
        )


def _subsample_pairs_stratified(
    pairs: list[tuple[str, str]],
    max_images: int,
    random_state: int = 42,
) -> list[tuple[str, str]]:
    """Cap training set size while keeping every class represented (avoids NORMAL-before-PNEUMONIA bias)."""
    if len(pairs) <= max_images:
        return pairs

    from collections import defaultdict

    rng = np.random.default_rng(random_state)
    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in pairs:
        by_label[item[1]].append(item)

    n_classes = len(by_label)
    min_take = min(2, min(len(v) for v in by_label.values()))
    if min_take * n_classes > max_images:
        min_take = max(1, max_images // n_classes)

    selected: list[tuple[str, str]] = []
    leftovers: list[tuple[str, str]] = []
    for lbl in sorted(by_label):
        items = by_label[lbl]
        order = rng.permutation(len(items))
        shuffled = [items[i] for i in order]
        selected.extend(shuffled[:min_take])
        leftovers.extend(shuffled[min_take:])

    remaining = max_images - len(selected)
    if remaining > 0 and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])

    rng.shuffle(selected)
    return selected[:max_images]


def _prepare_extract_dir(extract_dir: Path) -> None:
    """Fresh ZIP extract so stale single-class folders cannot linger between runs."""
    if extract_dir.exists():
        shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)


def _collect_images(root: Path) -> list[tuple[str, str]]:
    """Return list of (path, label) from folder structure or filename patterns."""
    if root.suffix.lower() == ".zip":
        return []

    path_pairs = _collect_by_path_labels(root)
    if _good_label_set(path_pairs):
        return path_pairs

    folder_pairs = _best_class_folder_pairs(root)
    if _good_label_set(folder_pairs):
        return folder_pairs

    shallow = _collect_from_class_folders(root)
    shallow_labels = {label for _, label in shallow}
    if shallow and len(shallow_labels) == 1 and next(iter(shallow_labels)) in GENERIC_FOLDER_NAMES:
        filename_pairs = _collect_from_filename_labels(root)
        if _good_label_set(filename_pairs, min_images=10):
            return filename_pairs

    # Return best effort for clearer validation errors
    for candidate in (path_pairs, folder_pairs, shallow, _collect_from_filename_labels(root)):
        if candidate:
            return candidate
    return []


class ImageModalityPipeline(BaseModalityPipeline):
    modality = "image"
    pipeline_type = "image_classification"

    def run(
        self,
        job_id: str,
        dataset_path: str,
        target_column: str,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        t0 = time.time()
        settings = get_settings()
        img_cfg = settings.modalities.get("image", {})
        size = tuple(img_cfg.get("resize", [64, 64]))
        max_images = int(img_cfg.get("max_images", 500))
        path = Path(dataset_path)

        self.report("preprocessing", "Loading and resizing images (RGB)", 18)
        extract_dir = settings.upload_dir / f"{metadata.get('dataset_id', job_id)}_{job_id}_images"

        if path.suffix.lower() == ".zip":
            _prepare_extract_dir(extract_dir)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(extract_dir)
            scan_root = _resolve_scan_root(extract_dir)
        else:
            scan_root = path.parent if path.is_file() else path
            if scan_root.is_dir():
                scan_root = _resolve_scan_root(scan_root)

        pairs = _collect_images(scan_root)
        if len(pairs) < 10:
            raise ValueError(
                "Need a ZIP or folder with class subfolders (e.g. dog/, cat/) or "
                "breed filenames (e.g. shiba_inu_29.jpg) with 10+ images."
            )

        n_classes = len({label for _, label in pairs})
        if n_classes < 2:
            raise ValueError(
                f"Need at least 2 image classes for classification, found 1: "
                f"{pairs[0][1] if pairs else 'unknown'}. "
                "Use folder-per-class (dog/, cat/) or breed filenames (shiba_inu_29.jpg)."
            )

        _validate_image_labels(pairs)

        pairs = _subsample_pairs_stratified(pairs, max_images)
        _validate_image_labels(pairs)

        from collections import Counter

        planned_counts = Counter(label for _, label in pairs)
        self.report(
            "preprocessing",
            f"Using {len(pairs)} images across {len(planned_counts)} classes: "
            + ", ".join(f"{k}={v}" for k, v in sorted(planned_counts.items())),
            25,
        )

        X_list, y_list = [], []
        for fp, label in pairs:
            vec = load_image_vector(Path(fp), size=size, rgb=True)
            if vec is not None:
                X_list.append(vec)
                y_list.append(label)
        if len(X_list) < 10:
            raise ImportError("Image training requires Pillow: pip install pillow")

        loaded_counts = Counter(y_list)
        if len(loaded_counts) < 2:
            raise ValueError(
                f"Training ended up with only one class ({dict(loaded_counts)}). "
                f"Planned class mix was {dict(planned_counts)}. "
                "Restart the server (close start.bat window, run again) and retrain."
            )

        X = np.vstack(X_list)
        y_raw = np.array(y_list)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)

        from app.utils.split_utils import stratify_vector

        self.report("preprocessing", "PCA dimensionality reduction", 35)
        n_components = min(128, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42, stratify=stratify_vector(y, test_size=0.2)
        )
        y_test_labels = label_encoder.inverse_transform(y_test)

        candidates = {
            "pca_lgbm": LGBMClassifier(
                n_estimators=400, learning_rate=0.05, class_weight="balanced", verbose=-1,
            ),
            "pca_xgboost": XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6, eval_metric="mlogloss",
                tree_method="hist", n_jobs=-1,
            ),
            "pca_hist_gbm": HistGradientBoostingClassifier(max_iter=300, random_state=42),
            "pca_random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42),
            "pca_logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        }

        self.report("training", "Training image classifiers (PCA + boosting)", 60)
        best_name, best_clf, best_f1 = "", None, -1.0
        leaderboard: dict[str, float] = {}
        for name, clf in candidates.items():
            try:
                clf.fit(X_train, y_train)
                pred = label_encoder.inverse_transform(clf.predict(X_test).astype(int))
            except Exception:
                continue
            f1 = float(f1_score(y_test_labels, pred, average="weighted", zero_division=0))
            leaderboard[name] = round(f1, 4)
            if f1 > best_f1:
                best_f1, best_name, best_clf = f1, name, clf

        if best_clf is None:
            raise ValueError("All image classifiers failed to train. Check labels and image count.")

        pred = label_encoder.inverse_transform(best_clf.predict(X_test).astype(int))
        metrics = {
            "accuracy": round(float(accuracy_score(y_test_labels, pred)), 4),
            "f1_weighted": round(float(f1_score(y_test_labels, pred, average="weighted", zero_division=0)), 4),
        }

        self.report("explainability", "PCA component summary", 85)
        out_dir = settings.artifacts_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pca": pca,
            "classifier": best_clf,
            "image_size": list(size),
            "rgb": True,
            "model_name": best_name,
            "label_classes": label_encoder.classes_.tolist(),
        }
        model_path = out_dir / "image_pipeline.joblib"
        joblib.dump(bundle, model_path)

        self.report("complete", "Image pipeline complete", 100)
        return {
            "job_id": job_id,
            "modality": self.modality,
            "pipeline_type": self.pipeline_type,
            "task_type": "classification",
            "model_name": best_name,
            "label_classes": label_encoder.classes_.tolist(),
            "class_counts": dict(loaded_counts),
            "metrics": metrics,
            "baseline_scores": leaderboard,
            "n_images": len(X_list),
            "explainability": {
                "method": "pca_components",
                "n_components": n_components,
                "note": "For production CNNs use ResNet/EfficientNet transfer learning",
            },
            "drift_notes": ["Pixel histogram drift", "Embedding centroid shift", "Concept drift"],
            "pipeline_path": str(model_path),
            "training_seconds": round(time.time() - t0, 1),
        }
