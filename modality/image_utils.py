"""Shared image loading utilities for training and prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def load_image_vector(path: Path, size: tuple[int, int] = (64, 64), rgb: bool = True) -> Optional[np.ndarray]:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(path)
        img = img.convert("RGB" if rgb else "L").resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten()
    except Exception:
        return None


def resolve_image_path(raw: str) -> Path:
    from app.core.config import BASE_DIR

    p = Path(raw.strip().strip('"'))
    if not p.is_absolute():
        candidate = (BASE_DIR / p).resolve()
        if candidate.exists():
            return candidate
    return p.expanduser().resolve()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS and path.is_file()
