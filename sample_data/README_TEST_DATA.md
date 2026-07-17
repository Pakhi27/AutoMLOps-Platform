# Test Data Guide — Multi-Modal Platform

Use these files from `sample_data/` to test each modality in the UI (Step 1 Upload).

**Kaggle datasets for every modality:** see [KAGGLE_DATASETS_BY_MODALITY.md](../docs/KAGGLE_DATASETS_BY_MODALITY.md)

## Quick reference

| Modality | File | Target column | Extra notes |
|----------|------|---------------|-------------|
| Tabular | `customer_churn_sample.csv` | `churn` | Classification |
| Tabular | `housing.csv` | `median_house_value` | Regression |
| Text | `text_reviews_sample.csv` | `sentiment` | Text column: `review_text` |
| Time-Series | `sales_timeseries_sample.csv` | `sales` | Datetime: `timestamp` |
| Logs/Tickets | `incident_logs_sample.csv` | `assignment_group` | Text: `message` |
| Documents | `sample_document.txt` | (auto) | Or any PDF |
| Image | `image_samples.zip` | (folder labels) | ZIP with class folders |
| Database | `sample_test.db` | `churn` | Use API connect (see below) |

---

## 1. Tabular (CSV)

**File:** `customer_churn_sample.csv`  
**Target:** `churn`  
**Steps:** Upload → Profile → EDA → Train (5 trials is enough for testing)

**File:** `housing.csv`  
**Target:** `median_house_value` (regression)

---

## 2. Text

**File:** `text_reviews_sample.csv`  
**Target:** `sentiment`  
**Expected modality:** `text`  
**Pipeline:** TF-IDF + classifier

---

## 3. Time-Series

**File:** `sales_timeseries_sample.csv`  
**Target:** `sales` or `revenue`  
**Expected modality:** `timeseries`  
**Datetime column:** `timestamp` (auto-detected)

---

## 4. Logs / Tickets

**File:** `incident_logs_sample.csv`  
**Target:** `assignment_group` (predict which team owns the ticket)  
**Expected modality:** `logs`  
**Alternative:** Use `resolution_group` as target

For **clustering only** (no target): upload and use a dummy target — or use rows without labels (platform clusters by message).

---

## 5. PDF / Documents

**File:** `sample_document.txt` (or any `.pdf`)  
**Expected modality:** `documents`  
**Target:** If CSV extracted, use label column; for single TXT use default

---

## 6. Image

**File:** `image_samples.zip`  
**Structure inside ZIP:**
```
ok/
  img_001.png
  img_002.png
defect/
  img_003.png
  img_004.png
```
**Expected modality:** `image`  
**Target:** folder name = class label (no column needed — use any placeholder if UI asks)

Generate ZIP: `py -3.11 scripts/create_image_sample_zip.py`

---

## 7. Database (API — no file upload)

**File:** `sample_test.db` (SQLite)

```bash
curl -X POST http://localhost:8000/multimodal/connect-database \
  -H "Content-Type: application/json" \
  -d "{\"connection_url\": \"sqlite:///./sample_data/sample_test.db\", \"table\": \"customers\"}"
```

Returns `dataset_id` — use that in Train step with target `churn`.

Create DB: `py -3.11 scripts/create_sample_database.py`

---

## External datasets (optional, larger tests)

| Use case | Source |
|----------|--------|
| Tabular | [Telco Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Text | [IMDB / Sentiment140](https://www.kaggle.com/datasets) |
| Image | [MNIST folders](https://www.kaggle.com/datasets) or CIFAR-10 as ZIP folders |
| Time-Series | [Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) |

---

## After upload — what to expect

| Detected badge | Skip to step |
|----------------|--------------|
| `tabular` | Profile → EDA → Train |
| `text`, `logs`, `timeseries`, `documents`, `image` | Train directly |

Hard refresh UI (`Ctrl+Shift+R`) if you don't see modality badges.
