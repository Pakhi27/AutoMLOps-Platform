from app.services.outlier_detector import OutlierCapTransformer


def test_outlier_capping_reduces_max(classification_df):
    X = classification_df.drop(columns=["churned"]).copy()
    X["income"] = X["income"].fillna(X["income"].median())
    detector = OutlierCapTransformer(method="iqr", iqr_multiplier=1.5)
    X_capped = detector.fit_transform(X)
    assert X_capped["income"].max() < X["income"].max()


def test_outlier_transformer_preserves_row_count(classification_df):
    X = classification_df.drop(columns=["churned"]).copy()
    X["income"] = X["income"].fillna(X["income"].median())
    detector = OutlierCapTransformer()
    X_out = detector.fit_transform(X)
    assert len(X_out) == len(X)


def test_outlier_report_has_counts(classification_df):
    X = classification_df.drop(columns=["churned"]).copy()
    X["income"] = X["income"].fillna(X["income"].median())
    detector = OutlierCapTransformer()
    detector.fit(X)
    report = detector.get_report()
    assert "income" in report["n_outliers_per_column"]
    assert report["n_outliers_per_column"]["income"] >= 1
