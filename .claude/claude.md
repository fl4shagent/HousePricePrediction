# HDB Resale Price Prediction

## Project Overview
ML pipeline to predict Singapore HDB resale flat prices. Trained on 170k+ transactions from data.gov.sg, enriched with geospatial features (MRT proximity, school proximity, building metadata). Final model is an LGBM + XGBoost ensemble served via FastAPI.

## Tech Stack
- **Language:** Python 3.10+
- **Data:** pandas, numpy, geopy, aiohttp, requests
- **ML:** scikit-learn, LightGBM, XGBoost
- **Visualization:** matplotlib, seaborn
- **API:** FastAPI, uvicorn
- **Serialization:** joblib
- **Environment:** Local Jupyter notebooks (VS Code)

## Workflow
1. **Data Collection** (`notebooks/0_data_collection.ipynb`) — Fetch HDB transactions, MRT stations, schools, building info, hawker centres, and car park info from data.gov.sg API
2. **EDA** (`notebooks/1_eda.ipynb`) — Explore distributions, correlations, time trends
3. **Feature Engineering** (`notebooks/2_feature_engineering.ipynb`) — Geocode buildings, compute MRT/school/hawker/mall distances, CBD distance, car park features, mature estate flag, merge building metadata, train-test split
4. **Model Building** (`notebooks/3_model_building.ipynb`) — Train 8 models, hyperparameter tune, ensemble, evaluate
5. **API** (`api/`) — FastAPI endpoint for real-time predictions

## Progress vs Inspiration (Reference Notebooks)

### What we keep from the reference
- data.gov.sg as primary data source
- OneMap API for geocoding buildings and schools
- Geospatial features: closest MRT distance, closest school by level, elite school flags
- Building metadata merge from HDB property info
- Temporal train-test split (no random split — prevents data leakage)
- Model progression: baselines → tree models → gradient boosting → ensemble
- Evaluation metrics: MAE, RMSE, MAPE, R², MdAE, PER10, Error by Price Quartile

### Changes from the reference
| Area | Reference (Inspiration) | Our Project | Reason |
|------|------------------------|-------------|--------|
| **Environment** | Google Colab | Local Jupyter (VS Code) | Local dev preference |
| **MLP model** | Included (TensorFlow) | Removed | Underperformed LGBM/XGBoost, heavy dependency |
| **Notebook structure** | 2 notebooks (EDA+FE combined, model) | 4 notebooks (collection, EDA, FE, model) | Cleaner separation of concerns |
| **Data collection** | Manual CSV download | Programmatic API fetch with retry | Reproducibility |
| **OneMap API calls** | No rate limiting, no caching | Semaphore(50) + checkpoint saves | Robustness for local dev |
| **MRT distance** | Static (all stations for all transactions) | Time-varying (filter by station opening date) | Static leaks future station info into historical transactions (e.g., TEL Stage 4 for 2020 sales) |
| **Unseen categories** | Manual imputation per category | `handle_unknown='ignore'` in OneHotEncoder | Cleaner, handles future unknowns |
| **Deployment** | AWS ECS + Vercel dashboard | Simple FastAPI endpoint | Scope is ML pipeline + simple API |
| **Macroeconomic features** | SORA, CPI, GDP, unemployment (197 features) | Not included | Transaction year/month captures time trends implicitly; revisit if model performance plateaus |
| **New spatial features** | Not in reference | CBD distance, mall distance, hawker distance, car park features, mature estate flag | Research showed these are top predictors in SG housing studies |
| **Reusable code** | All logic in notebooks | Extracted to `src/` modules | Reusability for API |

### Feature Engineering Tracker
| Feature | Reference | Our Project | Status |
|---------|-----------|-------------|--------|
| Closest MRT distance (km) — time-varying | ✅ (static) | ✅ (time-varying) | Planned — filter stations by opening date per transaction to prevent leakage |
| Closest MRT station name — time-varying | ✅ (static) | ✅ (time-varying) | Planned — uses only stations open at transaction date |
| Elite school flags (pri/sec/mixed) | ✅ | ✅ | Planned |
| Building info (max_floor, year_completed) | ✅ | ✅ | Planned |
| Facility flags (residential, commercial, etc.) | ✅ | ✅ | Planned |
| Storey range → numeric median | ✅ | ✅ | Planned |
| Remaining lease → months | ✅ | ✅ | Planned |
| Lat/Lng coordinates | ✅ | ✅ | Planned |
| CBD distance (Raffles Place) | ❌ | ✅ | Planned — #2-3 most important feature in published SG studies |
| Shopping mall distance | ❌ | ✅ | Planned — ~$4K MAE reduction documented; curated list of ~80-100 malls |
| Hawker centre distance | ❌ | ✅ | Planned — NEA GeoJSON on data.gov.sg; uniquely Singaporean factor |
| Car park features (type, lots) | ❌ | ✅ | Planned — HDB Carpark Info dataset; car park type encodes estate quality |
| Mature estate flag | ❌ | ✅ | Planned — static lookup (27 towns → binary); quick win |
| Macroeconomic indicators (SORA, CPI, GDP) | ✅ | ❌ | Deferred — transaction year/month captures trends implicitly; revisit if model plateaus |
| Cooling measure flags | ✅ | ❌ | Deferred — requires manual curation of policy dates; revisit if model plateaus |

### Model Decision Tracker
| Model | Reference Result | Our Target | Status | Notes |
|-------|-----------------|------------|--------|-------|
| Linear Regression | MAE ~51k | Baseline | Planned | |
| Ridge | MAE ~50k | Baseline+ | Planned | |
| Lasso | MAE ~50k | Baseline+ | Planned | |
| ElasticNet | MAE ~50k | Baseline+ | Planned | |
| Decision Tree | MAE ~60k | Comparison | Planned | Expected to overfit |
| Random Forest | MAE ~40k | Mid-tier | Planned | |
| LightGBM | MAE ~28k | Target <30k | Planned | GridSearchCV tuning |
| XGBoost | MAE ~27.6k | Target <30k | Planned | GridSearchCV tuning |
| LGBM+XGB Ensemble | MAE ~27.3k | Target <28k | Planned | Simple average |
| MLP (TensorFlow) | MAE ~51k | — | Dropped | No improvement over baselines |

### Validation Strategy
| Decision | Choice | Reason |
|----------|--------|--------|
| **CV Method** | Walk-Forward Backtesting (Expanding Window) | K-Fold leaks future data, inflates metrics by 20-40%. Walk-forward mimics real deployment |
| **Min training window** | 18 months | Enough data for stable model training (2017-01 → 2018-06) |
| **Test window** | 6 months per fold | Balances granularity vs number of folds |
| **Gap** | 1 month | Prevents autocorrelation leakage at train/test boundary |
| **Tuning folds** | 5 | Runtime constraint during GridSearchCV |
| **Final eval folds** | 10 | Full evaluation after best hyperparameters found |
| **Fold boundaries** | Date-based (not row-count) | Transaction volume varies by month; row-count splits create unequal time periods |
| **Metrics** | MAE, RMSE, MAPE, R², MdAE, PER10, Error by Quartile | MAE primary; MdAE robust to outliers; PER10 is industry AVM standard (target >80%); quartile breakdown checks fairness |
| K-Fold | Rejected | Trains on future to predict past — categorically wrong for temporal data |
