"""Safe train/test and cross-validation splits for classification."""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def min_class_count(y) -> int:
    series = pd.Series(y)
    if series.empty:
        return 0
    counts = series.value_counts()
    return int(counts.min()) if not counts.empty else 0


def validate_classification_target(y, min_per_class: int = 2) -> None:
    """Raise a clear error before training if any class is too small."""
    series = pd.Series(y)
    counts = series.value_counts()
    n_rows = len(series.dropna())
    n_classes = len(counts)

    if n_classes < 2:
        raise ValueError(
            "Classification needs at least 2 classes in the target column. "
            f"Found {n_classes} unique value(s)."
        )

    # Almost every row is its own class → user picked an ID/index column, not a label
    if n_rows >= 10 and n_classes / n_rows > 0.5:
        sample = ", ".join(str(c) for c in counts.index[:5].tolist())
        raise ValueError(
            f"Target column looks like an ID or row index, not a label: {n_classes} unique values "
            f"in {n_rows} rows (examples: {sample}…). "
            "Pick a real label column instead — e.g. churn, Survived, category, sentiment — "
            "not PassengerId, customer_id, or row numbers."
        )

    tiny = counts[counts < min_per_class]
    if not tiny.empty:
        if len(tiny) > 8:
            preview = ", ".join(f"'{idx}' ({int(n)} row{'s' if n != 1 else ''})" for idx, n in tiny.head(8).items())
            raise ValueError(
                f"Target has {len(tiny)} class(es) with too few examples (e.g. {preview}…). "
                f"Each class needs at least {min_per_class} rows. "
                "If you see hundreds of numeric classes with 1 row each, you likely chose an ID column — "
                "pick a different target (e.g. churn, category)."
            )
        details = ", ".join(f"'{idx}' ({int(n)} row{'s' if n != 1 else ''})" for idx, n in tiny.items())
        raise ValueError(
            f"Target has class(es) with too few examples for training: {details}. "
            f"Each class needs at least {min_per_class} rows. "
            "Pick a different target column, merge rare categories, or use more data."
        )


def stratify_vector(y, test_size: float = 0.2, min_train_per_class: int = 2) -> Optional[np.ndarray]:
    """Return y for stratified train_test_split, or None when stratification is unsafe."""
    if min_class_count(y) < 2:
        return None
    counts = pd.Series(y).value_counts()
    train_floor = (counts * (1.0 - test_size)).apply(np.floor).astype(int)
    if int(train_floor.min()) < min_train_per_class:
        return None
    return np.asarray(y)


def cross_validator(
    y,
    cv_folds: int,
    task_type: str,
    random_state: int = 42,
) -> tuple[Union[int, Any], int]:
    """Pick StratifiedKFold when possible; otherwise fall back to KFold."""
    if task_type != "classification":
        n_splits = max(2, cv_folds)
        return n_splits, n_splits

    min_count = min_class_count(y)
    n_splits = max(2, min(cv_folds, min_count))
    if min_count >= n_splits:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_splits
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_splits
