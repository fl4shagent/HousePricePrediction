# Singapore HDB Resale Price Prediction

An end-to-end machine learning pipeline that predicts Singapore HDB resale flat prices. Trained on **667,327 transactions spanning 25 years (2000–2025)** across all 26 HDB towns, the model predicts resale prices for any HDB flat given its location, size, lease, and market conditions.

### What it predicts
Given a flat's details (town, flat type, floor area, storey, remaining lease, location), the model outputs a **predicted resale price in SGD** — accurate to within **$26,000 on average** (3.9% error), with **94% of predictions landing within ±10% of the actual sale price**.

### Scale

| | |
|---|---|
| **Training data** | 667,327 transactions (Jan 2000 – Aug 2025) |
| **Test data** | 18,827 transactions (Sep 2025 – Jun 2026) |
| **Coverage** | All 26 HDB towns, 7 flat types, ~9,800 unique buildings |
| **Features** | 32 engineered features from 10 government data sources |
| **Model** | LightGBM + XGBoost ensemble with walk-forward backtesting |

### Final Model Performance (holdout test set, 2025-09 to 2026-06)

| Metric | Value | Meaning |
|--------|-------|---------|
| MAE | **$26,044** | Average prediction error |
| MAPE | **3.9%** | Average percentage error |
| PER10 | **94.2%** | Predictions within ±10% of actual |
| R² | **0.9693** | Variance explained |

---

## Table of Contents

1. [Data Collection](#1-data-collection)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Feature Engineering](#3-feature-engineering)
4. [Model Building & Evaluation](#4-model-building--evaluation)
5. [Model Comparison](#5-model-comparison)
6. [Tech Stack](#6-tech-stack)
7. [Project Structure](#7-project-structure)
8. [How to Run](#8-how-to-run)

---

## 1. Data Collection

All data comes from **official Singapore government sources** to ensure reliability and reproducibility.

| Dataset | Source | Records | Purpose |
|---------|--------|---------|---------|
| HDB Resale Transactions | [data.gov.sg](https://data.gov.sg) (4 API endpoints) | 692,719 | Target variable + base features |
| MRT/LRT Stations | [LTA DataMall](https://datamall.lta.gov.sg) (Shapefile, Mar 2026) | 190 | Nearest MRT distance |
| Station Opening Dates | LTA press releases / Wikipedia | 190 | Time-varying MRT feature |
| Schools | [data.gov.sg](https://data.gov.sg) | 337 | Elite school proximity |
| HDB Property Info | [data.gov.sg](https://data.gov.sg) | 13,289 | Building metadata |
| Hawker Centres | [data.gov.sg](https://data.gov.sg) (GeoJSON) | 129 | Hawker proximity |
| HDB Car Parks | [data.gov.sg](https://data.gov.sg) | 2,266 | Car park type feature |
| Shopping Malls | Curated from public data | 87 | Mall proximity |
| SORA 3M Interest Rate | MAS public records | 318 months | Macroeconomic feature |
| CPI (All Items) | [data.gov.sg](https://data.gov.sg) / SingStat | 780+ months | Macroeconomic feature |

**Key design decisions:**
- **4 historical transaction datasets** were concatenated (2000-2012, 2012-2014, 2015-2016, 2017-onwards) to maximize training data coverage from 2000 to 2026.
- **MRT coordinates from LTA official shapefile** — replaced initial training-data-generated coordinates after comparison showed 9 stations were >200m off (worst: Tanjong Katong at 666m error). Coordinates are converted from SVY21 polygon centroids to WGS84 lat/lng.
- **All API calls include retry with exponential backoff** and skip-if-already-downloaded caching to handle data.gov.sg 429 rate limiting.

---

## 2. Data Preprocessing

Raw transaction data requires significant cleaning before feature engineering.

### Data Quality Issues Resolved

| Issue | Scope | Resolution |
|-------|-------|------------|
| `remaining_lease` nulls | 421,854 rows (61%), all 2000-2014 | Computed from `lease_commence_date + 99 - transaction_year` (mean error 0.3 months) |
| `remaining_lease` format inconsistency | 3 formats across periods | Built a 3-way parser: null → compute, numeric string `"70"` → months, `"61 years 04 months"` → parse |
| Exact duplicate rows | 1,101 rows | Dropped |
| 2012 storey_range format | 6,838 rows use 5-floor bands instead of 3-floor | Median formula handles both formats identically |
| `lease_commence_date` collinearity | r=0.976 with remaining_lease | Dropped — remaining_lease is the economically meaningful feature |

### Preprocessing Steps

1. **Drop exact duplicates** — 1,101 rows removed
2. **Parse remaining_lease** → total months (3-format handler)
3. **Parse storey_range** → numeric median (`"10 TO 12"` → `11.0`)
4. **Extract transaction_year and transaction_month** from date
5. **Drop lease_commence_date** (collinear with remaining_lease, r=0.976)
6. **Convert Y/N columns** → boolean (residential, commercial, market_hawker, etc.)
7. **Drop rows with missing geospatial or property data** — 5,464 rows (0.8% of dataset)

### Rows Dropped Summary

| Step | Rows Before | Rows Dropped | % Lost | Reason |
|------|-------------|-------------|--------|--------|
| Exact duplicates | 692,719 | 1,101 | 0.16% | Data ingestion errors |
| Failed geocoding (no lat/lng) | 691,618 | 3,790 | 0.55% | OneMap API couldn't locate ~106 demolished/renamed buildings |
| Missing HDB property info | 691,618 | 5,464 | 0.79% | Buildings not in HDB property database (some overlap with geocoding failures) |
| **Total after cleanup** | 692,719 | **~6,500** | **0.94%** | **686,154 rows retained (99.1% of original data)** |

The 0.94% data loss is acceptable — the failed buildings are primarily demolished blocks from the 2000s that no longer exist in current databases. All remaining rows have complete features with zero nulls.

### Macroeconomic Edge Nulls

| Issue | Rows | Resolution |
|-------|------|------------|
| SORA null for Jan 2000 | 2,333 | Backfilled from Feb 2000 value (lag needs Dec 1999 which doesn't exist) |
| CPI null for recent months | 1,529 | Forward-filled from last published month (not yet released by SingStat) |

---

## 3. Feature Engineering

32 features across 5 categories, computed from 10 external datasets.

### Feature Summary

| Category | Feature | Source | Rationale |
|----------|---------|--------|-----------|
| **Structural** | `floor_area_sqm` | Transaction data | Strongest price predictor (r=0.48) |
| | `storey_median` | Transaction data | Higher floors = higher prices |
| | `remaining_lease_months` | Computed | Longer lease = more value |
| | `flat_type`, `flat_model` | Transaction data | Flat configuration affects price |
| | `town` | Transaction data | Location premium varies 2-3x across towns |
| **Building** | `max_floor_lvl` | HDB Property Info | Building height correlates with newer developments |
| | `year_completed` | HDB Property Info | Newer buildings command premiums |
| | `total_dwelling_units` | HDB Property Info | Block size affects pricing |
| | Facility flags (6) | HDB Property Info | Residential, commercial, market/hawker, etc. |
| **Spatial** | `dist_to_nearest_mrt_km` | LTA + OneMap API | **Time-varying** — only uses stations open at transaction date |
| | `closest_mrt` | LTA + OneMap API | Station name captures line-specific premiums |
| | `dist_to_cbd_km` | Haversine to Raffles Place | #2-3 most important feature in SG housing studies |
| | `dist_to_nearest_mall_km` | Curated mall list | ~$4K MAE reduction documented in literature |
| | `dist_to_nearest_hawker_km` | NEA GeoJSON | Uniquely Singaporean cultural factor |
| | `nearest_carpark_type` | HDB Carpark Info (SVY21→WGS84) | Car park type encodes estate quality |
| | `nearest_carpark_dist_km` | HDB Carpark Info | Parking accessibility |
| | `latitude`, `longitude` | OneMap API geocoding | Raw spatial position for tree models |
| **School** | Elite school flags (3) | MOE schools data | Primary, secondary, mixed level — SAP/autonomous/gifted/IP classification |
| | `is_mature_estate` | Static town lookup | 91% of million-dollar HDB deals in mature estates |
| **Macro** | `sora_3m_lagged` | MAS public records | Interest rates directly affect mortgage costs and demand |
| | `cpi_lagged` | SingStat | Inflation proxy — captures cost environment |
| **Temporal** | `transaction_year`, `transaction_month` | Transaction data | Captures seasonal and trend patterns |

### Key Feature Engineering Decisions

**Time-varying MRT distances:** For each transaction, we only consider MRT stations that were **open at the transaction date**. A flat sold in 2020 near Marine Parade was ~3km from the nearest MRT. Using today's station list would show ~0.3km (TEL Stage 4 opened June 2024), creating data leakage of ~$50-70K per flat. We maintain a station opening date lookup and filter per transaction.

**1-month lagged macroeconomic features:** For a transaction in March 2024, we use February 2024's SORA and CPI values. This prevents leakage — March data isn't available when March transactions occur.

**Geospatial feature caching:** Computing distances for ~10,000 buildings x 190 MRT stations x 318 months takes ~30 minutes. All intermediate results are cached to `data/interim/`:

| Cache File | What | First Run | Subsequent |
|------------|------|-----------|------------|
| `building_geocodes.csv` | Lat/lng per building | ~50 min (API) | instant |
| `mrt_distances.csv` | MRT distance per (building, month) | ~80 min | instant |
| `school_flags.csv` | Elite flags per building | ~5 min | instant |
| `spatial_distances.csv` | CBD/mall/hawker per building | ~5 min | instant |
| `carpark_features.csv` | Carpark type per building | ~5 min | instant |

This caching strategy reduces notebook re-run time from **~120 minutes to under 5 minute**, critical for iterating on model experiments without re-computing stable geospatial features.

---

## 4. Model Building & Evaluation

### Models Trained

8 regression models were trained and evaluated, progressing from simple baselines to gradient boosting ensembles:

| Model | Type | Purpose |
|-------|------|---------|
| Linear Regression | Linear | Baseline |
| Ridge (RidgeCV) | Linear + L2 | Regularized baseline |
| Lasso (LassoCV) | Linear + L1 | Feature selection baseline |
| ElasticNet | Linear + L1/L2 | Combined regularization |
| Decision Tree | Tree | Simple tree baseline |
| Random Forest | Ensemble | Bagged trees |
| LightGBM | Gradient Boosting | Primary model |
| XGBoost | Gradient Boosting | Primary model |

**Final model: LGBM + XGBoost ensemble** (simple average of predictions).

### Validation Strategy: Walk-Forward Backtesting

Standard K-fold cross-validation is **invalid for time-series data** — it trains on future data to predict the past, inflating metrics by 20-40%. We use **walk-forward backtesting** with expanding windows:

- **Minimum training window:** 18 months
- **Test window:** 6 months per fold
- **Gap:** 1 month between train/test (prevents autocorrelation leakage)
- **Folds:** 44 total (5 for hyperparameter tuning, 10 for final evaluation)
- **Fold boundaries:** Date-based, not row-count-based (transaction volume varies 2-3x across years)

### Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **MAE** | Average absolute error in SGD — primary metric |
| **RMSE** | Penalizes large errors more than MAE |
| **MAPE** | Scale-free percentage error |
| **R²** | Proportion of variance explained |
| **MdAE** | Median error — robust to outliers |
| **PER10** | % of predictions within ±10% of actual — industry AVM standard |

### Hyperparameter Tuning

RandomizedSearchCV with 50 iterations and 5 walk-forward folds for both LightGBM and XGBoost. Total tuning time: ~104 minutes.

---

## 5. Model Comparison

### v2 (No Macro Features) vs v3 (With SORA + CPI)

Both models trained on 2000-2025-08, tested on 2025-09 to 2026-06.

| Metric | v2 (no macro) | v3 (with macro) | Impact |
|--------|-------------|----------------|--------|
| **Ensemble MAE** | $26,000 | $26,044 | Flat — tree models already capture temporal patterns |
| **MAPE** | 4.0% | 3.9% | Slight improvement |
| **PER10** | 94.0% | 94.2% | Slight improvement |
| **Linear Reg MAE** | $82,184 | $69,964 | **-15% — macro features dramatically help linear models** |

**Insight:** Gradient boosting models learn temporal patterns from `transaction_year/month` alone. Macro features (SORA, CPI) provide the most value for **linear models** that can't learn non-linear time dependencies, and for **long-horizon predictions** where the model must extrapolate beyond its training period.

### All Models — Final Results (v3, Holdout Test Set)

| Model | MAE | RMSE | MAPE | R² | PER10 |
|-------|-----|------|------|-----|-------|
| **XGBoost** | **$25,786** | $36,704 | 3.9% | 0.9701 | 94.4% |
| **Ensemble** | **$26,044** | $37,174 | 3.9% | 0.9693 | 94.2% |
| LightGBM | $27,600 | $39,229 | 4.1% | 0.9658 | 93.4% |
| Random Forest | $31,014 | $44,469 | 4.5% | 0.9561 | 91.6% |
| Decision Tree | $36,524 | $52,548 | 5.4% | 0.9387 | 85.9% |
| Linear Regression | $69,964 | $100,292 | 10.7% | 0.7766 | 59.2% |
| Ridge | $69,965 | $100,293 | 10.7% | 0.7766 | 59.2% |
| Lasso | $74,475 | $107,121 | 11.3% | 0.7452 | 56.8% |
| ElasticNet | $277,548 | $344,729 | 36.9% | -1.6391 | 10.3% |

### Comparison with Reference Project

| Metric | Reference Project | Our Model |
|--------|------------------|-----------|
| **MAE** | ~$27,000 | **$26,044** |
| **RMSE** | ~$39,000 | **$37,174** |
| **MAPE** | ~5.7% | **3.9%** |
| Training data | 232k rows (2017+) | 667k rows (2000+) |
| Features | 197 (incl. GDP, unemployment) | 32 (focused set) |
| MRT distances | Static (all stations) | Time-varying (by opening date) |

### Error by Price Quartile

| Quartile | Count | MAE | MAPE | PER10 |
|----------|-------|-----|------|-------|
| Q1 (cheapest) | 4,724 | $18,788 | 4.6% | 90.2% |
| Q2 | 4,830 | $20,006 | 3.5% | 96.3% |
| Q3 | 4,582 | $22,874 | 3.3% | 97.3% |
| Q4 (expensive) | 4,691 | $42,664 | 4.4% | 93.0% |

---

## 6. Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Geospatial | geopy, geopandas, pyproj |
| ML | scikit-learn, LightGBM, XGBoost |
| Visualization | matplotlib, seaborn |
| API | FastAPI, uvicorn |
| External APIs | OneMap (geocoding), data.gov.sg, LTA DataMall |

---

## 7. Project Structure

```
HousePricePrediction/
├── notebooks/
│   ├── 0_data_collection.ipynb      # Fetch all raw datasets
│   ├── 1_eda.ipynb                  # Exploratory data analysis
│   ├── 2_feature_engineering.ipynb  # Feature engineering + preprocessing
│   ├── 3_model_building.ipynb       # Train, evaluate, compare models
│   └── 4_hyperparameter_tuning.ipynb
├── src/
│   ├── data_collection.py           # API fetch, MRT stations, mall list
│   ├── feature_engineering.py       # Geocoding, distance computation
│   └── preprocessing.py             # Parsing, type conversion
├── data/
│   ├── raw/                         # Downloaded CSVs (gitignored)
│   ├── interim/                     # Cached geospatial computations
│   └── processed/                   # Final train/test splits
├── models/
│   ├── v2_resplit/                  # Without macro features
│   ├── v3_macro/                    # With SORA + CPI
│   └── *.joblib                     # Latest model artifacts
├── api/                             # FastAPI prediction endpoint
├── results_baseline.md              # Baseline results for comparison
├── progress.md                      # Development log
└── roadmap.md                       # Future improvements
```

---

## 8. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
# 0: Data collection (fetches from APIs, ~5 min first run)
# 1: EDA (read-only analysis)
# 2: Feature engineering (~75 min first run, <1 min with cache)
# 3: Model building (~10 min)
# 4: Hyperparameter tuning (~100 min, optional)
```

Notebooks are numbered and must be run in order. All intermediate computations are cached to `data/interim/` — subsequent runs are fast.
