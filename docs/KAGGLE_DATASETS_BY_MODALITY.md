# Kaggle Datasets by Modality — Testing Guide

Use these public Kaggle datasets to test each pipeline in the AutoMLOps platform.  
Download from [kaggle.com/datasets](https://www.kaggle.com/datasets) (free account + Kaggle API or browser download).

---

## 1. Tabular

| Dataset | Kaggle slug | Task | Target column | Notes |
|---------|-------------|------|---------------|-------|
| **Telco Customer Churn** | `blastchar/telco-customer-churn` | Classification | `Churn` | Classic churn; great for EDA + Optuna |
| **Titanic** | `yasserh/titanic-dataset` | Classification | `Survived` | Small, fast smoke test |
| **House Prices** | `c/house-prices-advanced-regression-techniques` | Regression | `SalePrice` | Many numeric + categorical features |
| **Credit Card Fraud** | `mlg-ulb/creditcardfraud` | Classification | `Class` | Imbalanced — tests class weights |
| **California Housing** | `camnugent/california-housing-prices` | Regression | `median_house_value` | Already have `housing.csv` sample |

**Upload:** `.csv` directly  
**Pipeline:** Tabular AutoML (XGBoost, LightGBM, CatBoost + Optuna)  
**Features that work:** Profile, EDA, Train, SHAP predict, Drift, Counterfactual, Advisor

---

## 2. Text

| Dataset | Kaggle slug | Task | Text column | Target |
|---------|-------------|------|-------------|--------|
| **IMDB Movie Reviews** | `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews` | Sentiment | `review` | `sentiment` |
| **Twitter US Airline Sentiment** | `crowdflower/twitter-airline-sentiment` | Sentiment | `text` | `airline_sentiment` |
| **Spam Email** | `uciml/sms-spam-collection-dataset` | Spam detection | `v2` (message) | `v1` (ham/spam) |
| **Amazon Fine Food Reviews** | `snap/amazon-fine-food-reviews` | Sentiment | `Text` | `Score` (binarize 1–2 vs 4–5) |

**Upload:** `.csv` with a long text column  
**Pipeline:** TF-IDF + Logistic / SVM / LightGBM / XGBoost (best picked automatically)  
**Target:** Pick sentiment/label column  
**Features that work:** Train, keyword explainability, predict (text in JSON), Advisor  
**Skip:** EDA charts, SHAP (use keyword explanations instead)

---

## 3. Image

| Dataset | Kaggle slug | Task | Format for platform |
|---------|-------------|------|---------------------|
| **Cats vs Dogs** | `salader/dogs-vs-cats` | Binary classification | Re-zip as `cat/*.jpg`, `dog/*.jpg` folders |
| **Intel Image Classification** | `puneet6060/intel-image-classification` | 6-class scenes | Already folder-per-class — zip `seg_train/` |
| **Chest X-Ray Pneumonia** | `paultimothymooney/chest-xray-pneumonia` | Medical binary | Zip `train/NORMAL/`, `train/PNEUMONIA/` |
| **Plant Disease** | `vipoooool/new-plant-diseases-dataset` | Multi-class | Zip one class folder per disease |
| **Fashion MNIST (as PNG)** | `zalando-research/fashionmnist` | 10-class | Export to folders or use small subset ZIP |

**Upload:** `.zip` with structure:
```
class_a/
  img001.jpg
class_b/
  img002.jpg
```
**Target:** `label` (placeholder — real labels = folder names)  
**Pipeline:** RGB resize + PCA + Random Forest / HistGBM / XGBoost / Logistic (best picked)  
**Features that work:** Train, metrics, Advisor  
**Limited:** JSON predict (image_path), SHAP, counterfactual

**Prep script tip (Cats vs Dogs):**
```bash
# After downloading, organize into two folders then zip
```

---

## 4. Time-Series

| Dataset | Kaggle slug | Task | Datetime col | Target |
|---------|-------------|------|--------------|--------|
| **Store Sales** | `competitions/store-sales-time-series-forecasting` | Forecast sales | `date` | `sales` (after merge) |
| **Daily Delhi Climate** | `sujeetkumar1998/daily-delhi-climate-data` | Regression | `date` | `meantemp` |
| **Electricity Consumption** | `robikibler/electricity-consumption-2015-2020` | Forecast | `Date` | `Consumption` |
| **Air Passengers (classic)** | Search "air passengers time series csv" | Forecast | `Month` | `Passengers` |
| **Retail Sales** | `manjeetsingh/retaildataset` | Forecast | `Date` | `Sales` |

**Upload:** `.csv` with date + numeric value columns  
**Target:** Numeric column to forecast (`sales`, `meantemp`, etc.)  
**Pipeline:** Lag features + Ridge / Random Forest / HistGBM / LightGBM / XGBoost  
**Features that work:** Train, lag importance, Advisor  
**Predict:** Pass engineered lag features in JSON (not raw date alone)

---

## 5. Logs / Tickets

| Dataset | Kaggle slug | Task | Text col | Target |
|---------|-------------|------|----------|--------|
| **IT Ticket Classification** | Search `helpdesk ticket classification` on Kaggle | Routing | `Description` | `Category` |
| **Stack Overflow Tags** | `stackoverflow/stackoverflow` | Tag prediction | `Title` + `Body` | top tag column |
| **Cybersecurity Logs** | `surajkumarramkumar/cybersecurity-intrusion-detection-dataset` | Alert class | log/message cols | label column |
| **Sample in repo** | `incident_logs_sample.csv` | Assignment group | `message` | `assignment_group` |

**Upload:** `.csv` with message + metadata columns  
**Pipeline:** TF-IDF + classifiers (same as text) or KMeans clustering if no valid target  
**Features that work:** Train, keyword explainability, predict, Advisor

---

## 6. Documents (PDF / TXT)

| Dataset | Kaggle slug | Task | Format |
|---------|-------------|------|--------|
| **ArXiv abstracts** | `Cornell-University/arxiv` | Topic classify | Export CSV: `abstract` + `categories` |
| **BBC News** | `gauravduttakiit/bbc-news` | Topic | CSV with `text` + `category` |
| **20 Newsgroups** | `c/20-newsgroups` or sklearn export | Topic | CSV text + label |
| **PDF contracts** | Search "pdf document classification kaggle" | Doc class | Single PDFs or CSV of extracted text |
| **Sample in repo** | `sample_document.txt` | RAG index demo | Single `.txt` upload |

**Upload:** `.txt`, `.pdf`, or `.csv` with text column  
**Target:** `label` for single files; real label column for CSV  
**Pipeline:** Text extraction → chunking (RAG) → TF-IDF classifier if labeled  
**Features that work:** Train, RAG chunks artifact, Advisor

---

## 7. Database

| Dataset | Source | Table | Target |
|---------|--------|-------|--------|
| **SQLite sample** | `sample_data/sample_test.db` | `customers` | `churn` |
| **Northwind** | Kaggle `juanmah/world-cup` / any SQL dump | varies | varies |
| **Brazilian E-Commerce** | `olistbr/brazilian-ecommerce` | Import to PostgreSQL/SQLite | `order_status` |

**Connect via API:**
```bash
curl -X POST http://localhost:8000/multimodal/connect-database \
  -H "Content-Type: application/json" \
  -d "{\"connection_url\": \"sqlite:///./sample_data/sample_test.db\", \"table\": \"customers\"}"
```

---

## Quick start checklist

| Modality | Start here | Size | Difficulty |
|----------|------------|------|------------|
| Tabular | Telco Churn | ~7K rows | Easy |
| Text | IMDB 50K (use 5K subset first) | Medium | Easy |
| Image | Intel Image Classification | ~25K images | Medium |
| Time-Series | Daily Delhi Climate | ~1.5K rows | Easy |
| Logs | `incident_logs_sample.csv` | Tiny | Easy |
| Documents | BBC News CSV | ~2K rows | Easy |
| Database | `sample_test.db` | Tiny | Easy |

---

## Download with Kaggle CLI

```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d blastchar/telco-customer-churn -p ./downloads --unzip
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews -p ./downloads --unzip
kaggle datasets download -d puneet6060/intel-image-classification -p ./downloads --unzip
```

---

## Model upgrades (platform defaults)

| Modality | Models tried (best wins) |
|----------|--------------------------|
| Tabular | XGBoost, LightGBM, CatBoost, RF + Optuna |
| Text | TF-IDF + Logistic, SVM, NB, **LightGBM, XGBoost** |
| Image | RGB 64×64 + PCA + **RF, HistGBM, XGBoost, Logistic** |
| Time-Series | Ridge, RF, HistGBM + **LightGBM, XGBoost** |
| Logs | Same as text (+ KMeans if unsupervised) |
| Documents | TF-IDF classifier + RAG chunk index |

See `GET /multimodal/capabilities` for live capability list.
