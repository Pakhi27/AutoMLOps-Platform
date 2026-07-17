"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 80, size=n).astype(float),
            "income": rng.exponential(scale=50000, size=n),
            "city": rng.choice(["NYC", "LA", "Chicago", "Houston"], size=n),
            "signup_date": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
            "churned": rng.choice(["yes", "no"], size=n, p=[0.3, 0.7]),
        }
    )
    # inject some missing values and an outlier
    df.loc[rng.choice(n, 15, replace=False), "income"] = np.nan
    df.loc[0, "income"] = 5_000_000
    return df


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 250
    x1 = rng.normal(50, 10, size=n)
    x2 = rng.normal(20, 5, size=n)
    noise = rng.normal(0, 5, size=n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "category": rng.choice(["A", "B", "C"], size=n),
            "target": 3 * x1 - 2 * x2 + noise,
        }
    )
    df.loc[rng.choice(n, 10, replace=False), "x1"] = np.nan
    return df
