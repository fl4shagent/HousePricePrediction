# Progress Log

## Phase 0: Project Scaffolding (Completed)
- Created directory structure: `data/{raw,interim,processed}`, `notebooks/`, `models/`, `api/`, `src/`
- Created `.gitignore` — excludes data/, models/, LTA shapefile, reference notebooks, Python/Jupyter artifacts
- Created `requirements.txt` — pandas, numpy, scikit-learn, lightgbm, xgboost, matplotlib, seaborn, geopy, geopandas, pyproj, aiohttp, requests, fastapi, uvicorn, joblib, jupyter
- Created `src/__init__.py` and `api/__init__.py`
- Created `.claude/claude.md` with project overview, tech stack, workflow, and tracking tables
- Created `.claude/agents/researcher.md` and `.claude/agents/data-analyst.md`

## Phase 1: Data Collection (Completed)
**Notebook:** `notebooks/0_data_collection.ipynb`
**Module:** `src/data_collection.py`

### Datasets Collected
| Dataset | Rows | Source | Notes |
|---------|------|--------|-------|
| HDB Resale Transactions | 692,719 | data.gov.sg (4 datasets: 2000-2012, 2012-2014, 2015-2016, 2017+) | Combined into single CSV |
| MRT/LRT Stations | 190 | LTA DataMall shapefile (coords) + public records (opening dates) | Includes TEL Stage 3/4, Punggol Coast, Hume |
| Schools | 337 | data.gov.sg | SAP/autonomous/gifted/IP indicators |
| HDB Property Info | 13,289 | data.gov.sg | Building metadata (max floors, year completed, facility flags) |
| Hawker Centres | 129 | data.gov.sg (GeoJSON) | Parsed from GeoJSON polygons to lat/lng centroids |
| HDB Car Park Info | 2,266 | data.gov.sg | SVY21 coordinates, car park type, decks |
| Shopping Malls | 87 | Curated list | Major SG malls with lat/lng |

### Key Decisions
- **Time-varying MRT distances:** Station dataset includes opening dates. Feature engineering will filter to only stations open at transaction time to prevent data leakage.
- **LTA shapefile for MRT coordinates:** Replaced training-data-generated coordinates with official LTA DataMall shapefile (March 2026). Comparison showed 9 stations were >200m off, worst being Tanjong Katong (666m).
- **Unbuilt stations (Bukit Brown, Founders' Memorial, Bocc):** Set to `2099-12-31` so they never appear in historical calculations.
- **4 historical datasets concatenated:** Extended coverage from 2017-only to 2000-2026 (692k rows vs original 174k).

### Bugs Fixed During Development
| Bug | Root Cause | Fix |
|-----|-----------|-----|
| 429 rate limiting | data.gov.sg throttles rapid API calls | Added retry with exponential backoff (30s/60s/90s) |
| geopandas not found | `pip install` in terminal installs to wrong Python than notebook kernel | Added `%pip install` cell at top of notebook |
| Hawker `data["data"]` is None | API returns None before download is ready | Changed to `(data.get("data") or {}).get("url")` with polling |
| Hawker GeoJSON parsed as CSV | Content-type check failed, fell through to `pd.read_csv()` | Changed to try-JSON-first: `try: resp.json()` then `except: pd.read_csv()` |
| All API cells re-fetch on every run | No caching logic | Added `os.path.exists()` skip for all API cells |

---

## Phase 2: Exploratory Data Analysis (Completed)
**Notebook:** `notebooks/1_eda.ipynb`

### Analysis Sections
1. **Load & Inspect** — shape (692k x 11), dtypes, nulls, duplicates, describe, categorical value counts
2. **Target Variable** — histogram + boxplot with mean/median, summary stats, skewness (1.11)
3. **Categorical Distributions** — flat type frequency + price boxplots, top/bottom 15 towns by median price, flat model frequency
4. **Numerical Features** — floor area vs price scatter, lease commence date trend, storey range distribution + scatter
5. **Correlation** — heatmap of all numeric features, sorted correlation with target
6. **Time Series** — monthly median price with volume overlay, price by top 6 towns, yearly median + YoY % change
7. **Key Takeaways** — 7 observations documented

### EDA Findings
- Resale price is right-skewed (mean > median), log-transform near-symmetric (skew -0.18)
- Floor area is strongest linear predictor (r=0.479 with price)
- Newer leases command higher prices
- Higher floors = higher prices
- Central/mature towns have much higher median prices
- Temporal trends are non-stationary — 4 distinct price regimes

---

## Data Analyst Deep-Dive Report (Completed)
Ran a thorough data quality and collinearity analysis before feature engineering.

### Data Quality Issues Found
| Issue | Scope | Resolution |
|-------|-------|------------|
| `remaining_lease` nulls | 421,854 rows (61%), all 2000-2014 | Compute from `lease_commence_date + 99 - transaction_year` |
| Exact duplicate rows | 1,101 rows | Drop |
| Near-duplicate rows | 15,436 (same flat specs + month, different price) | Keep — legitimate multiple units |
| 2012 storey_range format | 6,838 rows use 5-floor bands instead of 3-floor | Parse normally, median still works |
| remaining_lease 3 formats | null / numeric string "70" / "X years Y months" | Build 3-way parser |
| 3 ROOM terrace outliers | 468 rows with 150-367 sqm | Keep — legitimate terrace flats |

### Collinearity Analysis
| Feature Pair | Correlation | Action |
|-------------|-------------|--------|
| `lease_commence_date` vs `remaining_lease_months` | r = 0.976 | **Drop `lease_commence_date`** — keep remaining_lease (economically meaningful) |
| `floor_area_sqm` vs `flat_type` | eta² = 0.906 | Keep both for tree models; consider dropping flat_type for linear models |
| All other pairs | r < 0.5 | No action needed |

### Distribution Findings
- **resale_price:** Log-normal (raw skew=1.11, log skew=-0.18). Log-transform for linear models.
- **Rare flat_types:** 1 ROOM (496), MULTI-GENERATION (272)
- **Rare flat_models:** 3Gen (77), Improved-Maisonette (87), Premium Maisonette (94) + 5 more < 500

### Price Regime Shifts
| Era | Median Price | Price/sqm | Context |
|-----|-------------|-----------|---------|
| 2000-2007 | $228k | ~$2,400 | Post-Asian crisis, flat market |
| 2008-2013 | $373k | ~$4,800 | Run-up, 2013 cooling measures |
| 2014-2019 | $410k | ~$4,300 | Cooling measures, prices stabilize |
| 2020-2026 | $538k | ~$6,500 | COVID rebound, 2.4x from 2000s |

### Action Items for Feature Engineering
1. Drop 1,101 exact duplicates
2. Parse remaining_lease with 3-format handler
3. Drop `lease_commence_date` after computing remaining_lease_months
4. Log-transform `resale_price` for linear models
5. Don't clip outliers
6. Handle 2012 storey_range 5-floor bands
7. Use date-based fold boundaries in walk-forward CV

---

## Phase 3: Feature Engineering & Preprocessing (Completed)
**Notebook:** `notebooks/2_feature_engineering.ipynb`
**Modules:** `src/feature_engineering.py`, `src/preprocessing.py`

### Block A: Data Cleaning
- Dropped 1,101 exact duplicates
- Parsed `remaining_lease` → months (3-format handler: null/numeric/"X years Y months")
- Parsed `storey_range` → numeric median
- Extracted `transaction_year` and `transaction_month`
- Dropped `lease_commence_date` (r=0.976 collinearity with remaining_lease)

### Block B: Geospatial Features
- **Geocoded 9,919 buildings** via OneMap API — 9,572 successful (96.5%), 347 failed
- **Time-varying MRT distance** — 560k unique (building, month) pairs × 190 stations
- **Geocoded 337 schools** — 326 by name, 11 by postal_code fallback, 337/337 total
- **Nearest school by level** (PRIMARY/SECONDARY/MIXED) + elite school flags
- **CBD distance** — haversine to Raffles Place
- **Nearest mall distance** — from 87 shopping malls
- **Nearest hawker centre distance** — from 129 hawker centres
- **Car park features** — SVY21→WGS84 conversion, matched nearest car park type

### Block C: Final Assembly
- Merged HDB property info (13,289 buildings, 5,464 unmatched)
- Added mature estate flag (15 mature towns)
- **Dropped 21,536 rows** (3.1%) with missing geo/property data
- Final: **670,082 rows × 31 columns, zero nulls**
- Train-test split: 598,942 train (2000-2023) / 71,140 test (2023-2026)

### Bugs Fixed
| Bug | Root Cause | Fix |
|-----|-----------|-----|
| 99.4% geocoding failure (9,860/9,919) | Concurrency=50 overwhelmed OneMap API, silent failures | Reduced to concurrency=10, batch_size=500, added retry with backoff |
| Schools 95% geocoding failure | Same concurrency issue | Same fix — reduced concurrency, added retry |
| Notebook cell had old concurrency=50 | Kernel cached old module after code fix | Updated notebook cell + documented kernel restart requirement |
| `libomp` missing for LightGBM on macOS | conda environment lacked OpenMP | `conda install llvm-openmp` |
| SORA nulls (2,333 rows) for Jan 2000 | 1-month lag needs Dec 1999 SORA which doesn't exist | `bfill` from Feb 2000 value |
| CPI nulls (1,529 rows) for recent months | CPI not yet published for most recent months | `ffill` from last published month |
| LinearRegression NaN crash | Macro nulls passed to sklearn which rejects NaN | Fixed by filling edge nulls before train-test split |

---

## Phase 4: Model Building — v2 (Completed)
**Notebook:** `notebooks/3_model_building.ipynb`
**Split:** Train 2000-2025-08 (667,327 rows) → Test 2025-09 to 2026-06 (18,827 rows)

### Model Results — Held-Out Test Set (2025-09 to 2026-06)

| Model | MAE | RMSE | MAPE | R² | PER10 |
|-------|-----|------|------|-----|-------|
| XGBoost | **$25,859** | $36,676 | 4.0% | 0.9701 | 94.0% |
| **Ensemble (LGBM+XGB)** | **$26,000** | **$36,750** | **4.0%** | **0.9700** | **94.0%** |
| LightGBM | $27,380 | $38,345 | 4.2% | 0.9673 | 92.8% |
| Random Forest | $29,434 | $42,657 | 4.4% | 0.9596 | 92.3% |
| Decision Tree | $36,683 | $53,114 | 5.5% | 0.9374 | 85.5% |
| Linear Regression | $82,184 | $116,299 | 11.3% | 0.6996 | 52.1% |
| Ridge | $82,184 | $116,301 | 11.3% | 0.6996 | 52.1% |
| Lasso | $84,914 | $122,932 | 11.6% | 0.6644 | 52.3% |
| ElasticNet | $279,290 | $346,286 | 37.1% | -1.6630 | 10.0% |

### Comparison with Reference Project
| Metric | Reference | Ours v2 | Result |
|--------|-----------|---------|--------|
| **MAE** | ~$27,000 | **$26,000** | We win |
| **RMSE** | ~$39,000 | **$36,750** | We win |
| **MAPE** | ~5.7% | **4.0%** | We win |
| **PER10** | — | **94.0%** | Above 80% industry target |

### Error by Price Quartile (Ensemble)
| Quartile | Count | MAE | MAPE | PER10 |
|----------|-------|-----|------|-------|
| Q1 (cheapest) | 4,724 | $19,798 | 4.8% | 89.4% |
| Q2 | 4,830 | $20,999 | 3.7% | 95.8% |
| Q3 | 4,582 | $22,835 | 3.3% | 97.3% |
| Q4 (expensive) | 4,691 | $40,486 | 4.2% | 93.6% |

### What Changed from v1 to v2
- **Split date moved:** 2023-08 → 2025-08 (model now trains on recent price levels)
- **More geocoded buildings:** 9,572 → 9,807 (retry picked up 235 previously failed)
- **Result:** MAE dropped from $60,637 to $26,000 (57% improvement)

### Artifacts Saved
- `models/v2_resplit/` — versioned copy
- `models/lgbm_model.joblib` (1,745 KB) — latest
- `models/xgb_model.joblib` (4,480 KB) — latest
- `models/preprocessor.joblib` (8 KB)
- `models/feature_config.json` (1 KB)