"""Small filesystem / IO helpers shared across services."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


def new_id(prefix: str = "") -> str:
    token = uuid.uuid4().hex[:12]
    return f"{prefix}{token}" if prefix else token


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV, trying a couple of common encodings before giving up."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    # last attempt, let pandas raise its natural error
    return pd.read_csv(path)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
