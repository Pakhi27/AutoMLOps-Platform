"""Tests for safe classification splits."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from app.utils.split_utils import (
    cross_validator,
    min_class_count,
    stratify_vector,
    validate_classification_target,
)


def test_min_class_count():
    y = pd.Series([0, 0, 1, 1, 1])
    assert min_class_count(y) == 2


def test_validate_classification_target_rejects_id_like_column():
    y = pd.Series(range(100))
    with pytest.raises(ValueError, match="looks like an ID"):
        validate_classification_target(y)


def test_validate_classification_target_rejects_singleton():
    y = pd.Series([0, 0, 0, 1])
    with pytest.raises(ValueError, match="too few examples"):
        validate_classification_target(y)


def test_stratify_vector_none_when_class_too_small():
    y = np.array([0, 0, 0, 1])
    assert stratify_vector(y, test_size=0.2) is None


def test_cross_validator_falls_back_when_class_singleton():
    y = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    cv, n_splits = cross_validator(y, cv_folds=5, task_type="classification", random_state=42)
    assert n_splits == 2
    assert not isinstance(cv, StratifiedKFold)
