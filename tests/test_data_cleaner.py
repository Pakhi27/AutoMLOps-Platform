import numpy as np

from app.services.data_cleaner import DataCleanerTransformer


def test_cleaner_imputes_missing_values(classification_df):
    X = classification_df.drop(columns=["churned"])
    cleaner = DataCleanerTransformer()
    X_clean = cleaner.fit_transform(X)
    assert X_clean.isna().sum().sum() == 0


def test_cleaner_preserves_row_count(classification_df):
    X = classification_df.drop(columns=["churned"])
    cleaner = DataCleanerTransformer()
    X_clean = cleaner.fit_transform(X)
    assert len(X_clean) == len(X)


def test_cleaner_drops_constant_columns(classification_df):
    X = classification_df.drop(columns=["churned"]).copy()
    X["constant_col"] = 1
    cleaner = DataCleanerTransformer()
    X_clean = cleaner.fit_transform(X)
    assert "constant_col" not in X_clean.columns


def test_cleaner_transform_matches_fit_columns(classification_df):
    X = classification_df.drop(columns=["churned"])
    cleaner = DataCleanerTransformer()
    cleaner.fit(X.iloc[:200])
    transformed = cleaner.transform(X.iloc[200:])
    assert not transformed.isna().any().any()
    assert len(transformed) == len(X.iloc[200:])
