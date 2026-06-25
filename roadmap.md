# Roadmap

## Completed

### Phase 0: Project Scaffolding ✅
### Phase 1: Data Collection ✅
### Phase 2: EDA ✅
### Phase 3: Feature Engineering & Preprocessing ✅
### Phase 4: Model Building (baseline) ✅
### Phase 4B: Hyperparameter Tuning (RandomizedSearchCV) ✅
- Tuned ensemble MAE: $60,131 (0.8% improvement over default $60,637)
- XGBoost gained most from tuning ($60,704 → $59,654)
- Total tuning time: ~104 minutes

---

## Current Sprint

### Phase 5: FastAPI Prediction API
**Status:** Next up
**Files:** `api/main.py`, `api/schemas.py`, `api/predict.py`

**Tasks:**
- [ ] `POST /predict` endpoint — accepts flat details, returns predicted price
- [ ] `GET /healthz` health check
- [ ] Load tuned LGBM + XGBoost models, average predictions
- [ ] Geospatial features via pre-computed lookup table (not live API)
- [ ] Test with curl

---

## Next Iteration: Macroeconomic Features
**Status:** After Phase 5
**Where:** `notebooks/2_feature_engineering.ipynb` (Block B6, before final assembly)

This is the highest-impact improvement available. Adding macro features and retraining gives us a direct before/after comparison with the current tuned model.

**Tasks:**
- [ ] Download SORA 3M from MAS (monthly CSV)
- [ ] Download CPI from SingStat (monthly)
- [ ] Merge by transaction month with 1-month lag (prevents leakage)
- [ ] Add cooling measure flags (manual date lookup)
- [ ] Re-run Block C (train-test split)
- [ ] Re-run model building notebook
- [ ] Re-run tuning notebook
- [ ] Compare: current model ($60,131 MAE) vs macro-enhanced model

**Expected impact:** High — would help capture the 2024-2026 price surge the model currently misses.

---

## Future Enhancements (Deferred)

### MRT Anticipation Effect
**Why deferred:** Current approach treats MRT premium as step function at opening. Conservative rather than leaky.
- [ ] Add feature: "new MRT station opening within 1km in next 2 years"
- [ ] Requires announced station dates + planned locations

### Additional Spatial Features
**Why deferred:** Lower expected impact than current feature set.
- [ ] Park/green space distance (NParks GeoJSON, documented 3% premium)
- [ ] Supermarket distance (SFA dataset on data.gov.sg)
- [ ] Hospital/polyclinic distance
- [ ] Bus stop density (LTA DataMall, requires API key)

### Model Improvements
- [ ] GridSearchCV with focused grid (thorough tuning, ~2-4 hours)
- [ ] Conformal prediction intervals (uncertainty quantification)
- [ ] Stacking ensemble (meta-learner on top of LGBM + XGBoost)
- [ ] CatBoost as additional ensemble member
- [ ] SHAP values for individual prediction explanations
- [ ] Log-transform target for linear models, compare with raw
- [ ] Shorter test set evaluation (2023-09 to 2024-09) for fairer reference comparison

### Deployment
**Why deferred:** Scope is ML pipeline + simple API for now.
- [ ] Containerize with Docker
- [ ] Deploy API to AWS ECS / Railway / Render
- [ ] Frontend dashboard (Next.js or Streamlit)
- [ ] Daily data refresh pipeline
- [ ] Model retraining schedule
- [ ] Monitoring and alerting (model drift detection)
