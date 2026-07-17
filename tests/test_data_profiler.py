from app.services.data_profiler import DataProfiler


def test_profile_basic_shape(classification_df):
    profile = DataProfiler().profile(classification_df)
    assert profile["n_rows"] == len(classification_df)
    assert profile["n_columns"] == len(classification_df.columns)
    assert "income" in profile["numeric_columns"]
    assert "city" in profile["categorical_columns"]


def test_profile_detects_missing_values(classification_df):
    profile = DataProfiler().profile(classification_df)
    assert profile["missing_values"]["income"]["n_missing"] > 0


def test_profile_detects_datetime_column(classification_df):
    profile = DataProfiler().profile(classification_df)
    assert "signup_date" in profile["datetime_columns"]
