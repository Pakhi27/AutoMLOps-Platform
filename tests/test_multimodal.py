"""Tests for multi-modal data type detection."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.services.modality.detector import DataModalityDetector


def test_detect_tabular():
    df = pd.DataFrame({"age": [1, 2, 3], "income": [10, 20, 30], "churn": ["yes", "no", "yes"]})
    meta = DataModalityDetector().detect_dataframe(df)
    assert meta["modality"] == "tabular"


def test_detect_text_column():
    df = pd.DataFrame({
        "review": ["this product is absolutely wonderful and amazing"] * 5 + ["terrible bad awful"] * 5,
        "sentiment": ["pos"] * 5 + ["neg"] * 5,
    })
    meta = DataModalityDetector().detect_dataframe(df)
    assert meta["modality"] == "text"
    assert meta.get("text_column") == "review"


def test_text_pipeline_multiclass_labels(tmp_path):
    from app.services.modality.text_pipeline import TextModalityPipeline

    csv_path = tmp_path / "tweets.csv"
    df = pd.DataFrame({
        "text": ["bad flight delay angry"] * 20 + ["okay average flight"] * 20 + ["great love this airline"] * 20,
        "airline_sentiment": ["negative"] * 20 + ["neutral"] * 20 + ["positive"] * 20,
    })
    df.to_csv(csv_path, index=False)

    result = TextModalityPipeline().run(
        job_id="job_text_mc",
        dataset_path=str(csv_path),
        target_column="airline_sentiment",
        metadata={"text_column": "text"},
    )
    assert result["task_type"] == "classification"
    assert set(result["label_classes"]) == {"negative", "neutral", "positive"}


def test_detect_airline_sentiment_as_text():
    """Twitter US Airline Sentiment must be text, not timeseries (user_timezone false positive)."""
    df = pd.DataFrame({
        "tweet_id": range(100),
        "text": ["United flight was delayed two hours terrible service"] * 50 + ["Virgin America great crew love it"] * 50,
        "airline_sentiment": ["negative"] * 50 + ["positive"] * 50,
        "retweet_count": [0] * 100,
        "airline_sentiment_confidence": [0.9] * 100,
        "user_timezone": ["Eastern Time (US & Canada)"] * 100,
        "tweet_created": ["2015-02-24 11:35:52 -0800"] * 100,
    })
    meta = DataModalityDetector().detect_dataframe(df)
    assert meta["modality"] == "text"
    assert meta.get("text_column") == "text"
    assert "airline_sentiment" in meta.get("suggested_targets", [])


def test_detect_timeseries():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=50, freq="D"),
        "sales": range(50),
    })
    meta = DataModalityDetector().detect_dataframe(df)
    assert meta["modality"] == "timeseries"


def test_detect_customer_churn_as_tabular():
    """Churn CSV has signup_date + numerics but is entity-level tabular, not forecasting."""
    path = Path(__file__).resolve().parents[1] / "sample_data" / "customer_churn_sample.csv"
    meta = DataModalityDetector().detect_file(path, path.name)
    assert meta["modality"] == "tabular"
    assert meta["pipeline_type"] == "tabular_automl"
    assert "churn" in meta.get("suggested_targets", [])


def test_detect_logs():
    df = pd.DataFrame({
        "message": ["error timeout", "connection reset"] * 10,
        "severity": ["high", "low"] * 10,
        "assignment_group": ["netops", "app"] * 10,
    })
    meta = DataModalityDetector().detect_dataframe(df)
    assert meta["modality"] == "logs"


def test_registry_capabilities():
    from app.services.modality.registry import get_modality_registry

    caps = get_modality_registry().capabilities()
    modalities = {c["modality"] for c in caps}
    assert "tabular" in modalities
    assert "text" in modalities
    assert "image" in modalities


def test_detect_txt_as_documents():
    meta = DataModalityDetector().detect_file(Path("sample.txt"), "sample.txt")
    assert meta["modality"] == "documents"
    assert "label" in meta.get("suggested_targets", [])


def test_image_predict_decodes_class_names(tmp_path):
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    import joblib

    from app.services.modality.predict_handlers import predict_modality

    # synthetic 64x64 RGB vectors for cat/dog
    rng = np.random.default_rng(42)
    X = np.vstack([rng.random(64 * 64 * 3) for _ in range(20)])
    y = np.array([0] * 10 + [1] * 10)
    pca = PCA(n_components=8, random_state=42)
    X_pca = pca.fit_transform(X)
    clf = LogisticRegression(max_iter=200).fit(X_pca, y)

    bundle = {
        "pca": pca,
        "classifier": clf,
        "image_size": [64, 64],
        "rgb": True,
        "label_classes": ["cat", "dog"],
    }
    artifact_path = tmp_path / "image.joblib"
    joblib.dump(bundle, artifact_path)

    # create a tiny png for predict
    try:
        from PIL import Image
    except ImportError:
        return
    img_path = tmp_path / "test_cat.png"
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(img_path)

    entry = {
        "job_id": "job_img_test",
        "modality": "image",
        "label_classes": ["cat", "dog"],
    }
    df = pd.DataFrame([{"image_path": str(img_path)}])
    result = predict_modality(entry, bundle, df)
    assert result["predictions"][0] in ("cat", "dog")
    assert result["image_rows"][0]["preview_url"].startswith("/predict/preview-image")
    probs = result["probabilities"][0]
    assert "cat" in probs and "dog" in probs


def test_image_collect_labels_from_breed_filenames(tmp_path):
    from app.services.modality.image_pipeline import _collect_images

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ["shiba_inu_1.jpg", "shiba_inu_2.jpg", "golden_retriever_1.jpg", "golden_retriever_2.jpg",
                 "beagle_1.jpg", "beagle_2.jpg", "poodle_1.jpg", "poodle_2.jpg",
                 "husky_1.jpg", "husky_2.jpg", "boxer_1.jpg", "boxer_2.jpg"]:
        (img_dir / name).write_bytes(b"fake")

    pairs = _collect_images(tmp_path)
    labels = {label for _, label in pairs}
    assert "images" not in labels
    assert "shiba_inu" in labels
    assert "golden_retriever" in labels
    assert len(labels) >= 2


def test_image_collect_dog_cat_folders(tmp_path):
    from app.services.modality.image_pipeline import _collect_images

    for cls in ("dog", "cat"):
        d = tmp_path / cls
        d.mkdir()
        for i in range(6):
            (d / f"{cls}_{i}.jpg").write_bytes(b"fake")

    pairs = _collect_images(tmp_path)
    labels = {label for _, label in pairs}
    assert labels == {"dog", "cat"}


def test_image_collect_nested_my_images_layout(tmp_path):
    from app.services.modality.image_pipeline import _collect_images

    root = tmp_path / "my_images"
    for cls in ("dog", "cat"):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(6):
            (d / f"img{i:03d}.jpg").write_bytes(b"fake")

    pairs = _collect_images(tmp_path)
    labels = {label for _, label in pairs}
    assert labels == {"dog", "cat"}
    assert len(pairs) == 12


def test_image_collect_train_dog_cat_layout(tmp_path):
    from app.services.modality.image_pipeline import _collect_images

    for cls in ("dog", "cat"):
        d = tmp_path / "train" / cls
        d.mkdir(parents=True)
        for i in range(6):
            (d / f"photo_{i}.jpg").write_bytes(b"fake")

    pairs = _collect_images(tmp_path)
    labels = {label for _, label in pairs}
    assert labels == {"dog", "cat"}
    assert len(pairs) == 12


def test_image_collect_deep_train_dog_cat_layout(tmp_path):
    from app.services.modality.image_pipeline import _collect_images

    for cls in ("dog", "cat"):
        d = tmp_path / "my_images" / "train" / cls
        d.mkdir(parents=True)
        for i in range(6):
            (d / f"img{i:03d}.jpg").write_bytes(b"fake")

    pairs = _collect_images(tmp_path)
    labels = {label for _, label in pairs}
    assert labels == {"dog", "cat"}
    assert len(pairs) == 12


def test_image_subsample_keeps_both_classes(tmp_path):
    from app.services.modality.image_pipeline import _collect_images, _subsample_pairs_stratified

    for cls in ("NORMAL", "PNEUMONIA"):
        d = tmp_path / "train" / cls
        d.mkdir(parents=True)
        for i in range(300):
            (d / f"img{i:04d}.jpg").write_bytes(b"fake")

    pairs = _collect_images(tmp_path)
    labels_all = {label for _, label in pairs}
    assert labels_all == {"normal", "pneumonia"}

    subsampled = _subsample_pairs_stratified(pairs, max_images=500)
    labels_sub = {label for _, label in subsampled}
    assert labels_sub == {"normal", "pneumonia"}
    assert len(subsampled) == 500
    from collections import Counter
    counts = Counter(l for _, l in subsampled)
    assert counts["normal"] > 0
    assert counts["pneumonia"] > 0


def test_image_rejects_numbered_one_image_per_class(tmp_path):
    from app.services.modality.image_pipeline import _collect_images, _validate_image_labels

    for i in range(12):
        d = tmp_path / str(i)
        d.mkdir()
        (d / "photo.jpg").write_bytes(b"fake")

    pairs = _collect_images(tmp_path)
    with pytest.raises(ValueError, match="ZIP layout issue"):
        _validate_image_labels(pairs)


def test_timeseries_pipeline_accepts_none_test_size(tmp_path):
    from app.services.modality.timeseries_pipeline import TimeSeriesModalityPipeline

    csv_path = tmp_path / "sales.csv"
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=50, freq="D"),
        "sales": range(50),
        "revenue": [x * 40 for x in range(50)],
    })
    df.to_csv(csv_path, index=False)

    result = TimeSeriesModalityPipeline().run(
        job_id="job_test_ts",
        dataset_path=str(csv_path),
        target_column="revenue",
        metadata={"datetime_column": "timestamp"},
        test_size=None,
    )
    assert result["task_type"] == "regression"
    assert "rmse" in result["metrics"]
