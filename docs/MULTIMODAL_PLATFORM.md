# Multi-Modal AutoMLOps Platform

## Unified Flow

```
Upload / Connect Data
        ↓
Auto-detect Data Type
        ↓
Select Pipeline Automatically
        ↓
Run Preprocessing
        ↓
Train Suitable Models
        ↓
Evaluate with Correct Metrics
        ↓
Generate Explainability
        ↓
Monitor Drift
        ↓
Create AI Advisor Report
```

---

## Supported Data Types

| Data Type | Formats | Use Cases |
|-----------|---------|-----------|
| **Tabular** | CSV, Excel (.xlsx/.xls), TSV, SQL tables | Classification, regression, feature engineering |
| **Text** | CSV with text column, .txt, .jsonl | Sentiment analysis, ticket classification, clustering, summarization |
| **Image** | ZIP folder (class/image.jpg), .jpg/.png | Image classification, defect detection, OCR (via document pipeline) |
| **Time-Series** | CSV with datetime + value columns | Forecasting, anomaly detection, trend prediction |
| **Logs/Tickets** | CSV with message + metadata columns | RCA, incident clustering, assignment group prediction |
| **PDF/Documents** | .pdf, .txt, CSV | Text/table extraction, document classification, RAG chatbot |
| **Database** | SQL via connection URL | Train directly from SQL tables |

---

## Per-Modality Pipeline Details

### 1. Tabular

| Stage | Details |
|-------|---------|
| **Preprocessing** | Missing imputation, outlier capping (IQR), one-hot/frequency encoding, log transforms, interactions, feature selection |
| **Models** | Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN + Optuna tuning |
| **Metrics** | Accuracy, F1, precision, recall, ROC-AUC (classification); R², RMSE, MAE (regression) |
| **Explainability** | SHAP global/local plots, feature importance, partial dependence |
| **Drift** | Evidently column drift, PSI, feature distribution shift |

### 2. Text

| Stage | Details |
|-------|---------|
| **Preprocessing** | Lowercasing, punctuation removal, tokenization, TF-IDF embeddings, n-gram features |
| **Models** | TF-IDF + Logistic Regression, TF-IDF + Linear SVM, Multinomial Naive Bayes; optional BERT/LLM classifiers |
| **Metrics** | Accuracy, F1, precision, recall, confusion matrix |
| **Explainability** | Top keywords per class, TF-IDF coefficient weights, SHAP on sparse features, attention highlights (deep models) |
| **Drift** | Vocabulary drift, embedding drift, term frequency shift |

**Example (text):**
- Preprocessing: cleaning, tokenization, embeddings
- Models: BERT, TF-IDF + ML, LLM classifiers
- Metrics: accuracy, F1, precision, recall
- Explainability: keywords, SHAP, attention highlights
- Drift: vocabulary drift, embedding drift

### 3. Image

| Stage | Details |
|-------|---------|
| **Preprocessing** | Resize, grayscale/normalize, optional augmentation; PCA on pixel features (baseline) |
| **Models** | PCA + Logistic Regression (baseline); production: CNN ResNet/EfficientNet, transfer learning |
| **Metrics** | Accuracy, F1, top-k accuracy, confusion matrix |
| **Explainability** | Grad-CAM, saliency maps, SHAP on embeddings |
| **Drift** | Pixel histogram drift, embedding centroid shift, concept drift |

**Upload format:** ZIP with folders `class_a/`, `class_b/` containing images.

### 4. Time-Series

| Stage | Details |
|-------|---------|
| **Preprocessing** | Datetime parsing, sorting, lag features (1,2,3,7,14), rolling mean/std, hour/day/month features |
| **Models** | Lag + Ridge, Random Forest, HistGradientBoosting; optional ARIMA/Prophet |
| **Metrics** | MAE, RMSE, MAPE, sMAPE, R² |
| **Explainability** | Lag feature importance, SHAP on engineered features, forecast residual analysis |
| **Drift** | Value distribution shift, seasonality change, anomaly rate drift |

### 5. Logs / Tickets

| Stage | Details |
|-------|---------|
| **Preprocessing** | Timestamp parsing, log parsing, tokenization, TF-IDF, severity normalization |
| **Models** | TF-IDF + classifier (supervised); KMeans clustering (unsupervised RCA) |
| **Metrics** | Accuracy, F1 (classification); silhouette proxy (clustering) |
| **Explainability** | Important tokens, cluster themes, RCA keyword groups |
| **Drift** | New error templates, volume spike, assignment pattern drift |

### 6. PDF / Documents

| Stage | Details |
|-------|---------|
| **Preprocessing** | PDF text extraction (pypdf), table extraction, chunking for RAG |
| **Models** | Document classifier (TF-IDF pipeline); RAG index for chatbot |
| **Metrics** | Classification accuracy; retrieval precision@k; ROUGE (summarization) |
| **Explainability** | Highlighted passages, retrieved chunks, citation-style evidence |
| **Drift** | Topic drift, vocabulary shift, retrieval quality decay |

### 7. Database

| Stage | Details |
|-------|---------|
| **Connect** | `POST /multimodal/connect-database` with SQLAlchemy URL |
| **Flow** | Export table → auto-detect modality → route to tabular/text/timeseries pipeline |
| **Examples** | `sqlite:///data.db`, `postgresql://user:pass@host:5432/dbname` |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/multimodal/capabilities` | List all modalities + preprocessing/models/metrics/drift |
| POST | `/multimodal/upload` | Upload any supported format with auto-detection |
| POST | `/multimodal/connect-database` | Connect SQL and import as dataset |
| GET | `/multimodal/datasets/{id}/modality` | Get detected modality for a dataset |
| POST | `/pipeline/run` | Train (auto-routes by modality); optional `modality`, `text_column`, `datetime_column` |

### Example: Database connect

```bash
curl -X POST http://localhost:8000/multimodal/connect-database \
  -H "Content-Type: application/json" \
  -d '{
    "connection_url": "sqlite:///./mydata.db",
    "table": "customers",
    "limit": 50000
  }'
```

### Example: Text pipeline training

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "ds_abc123",
    "target_column": "sentiment",
    "text_column": "review_text",
    "modality": "text"
  }'
```

---

## Architecture

```
app/services/modality/
├── detector.py          # Auto-detect data type
├── registry.py          # Route modality → pipeline
├── tabular_pipeline.py  # Existing 14-stage AutoML
├── text_pipeline.py     # TF-IDF + sklearn NLP
├── timeseries_pipeline.py
├── logs_pipeline.py
├── document_pipeline.py
├── image_pipeline.py
└── db_connector.py      # SQL → dataset
```

Metadata stored at: `data/uploads/{dataset_id}.meta.json`

---

## Enterprise Value

This enhancement makes the platform more enterprise-ready by allowing users to solve multiple ML use cases from one system instead of being limited to CSV-based tabular models. It supports structured, semi-structured, and unstructured data with automated pipeline selection, model training, explainability, monitoring, and AI-generated recommendations.

---

## Optional Deep-Learning Extensions

For production image (CNN) or text (BERT) models, install optionally:

```bash
pip install torch transformers torchvision
```

The platform uses lightweight sklearn baselines by default so it runs without GPU dependencies.
