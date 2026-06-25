# Model Results — v2 (Train: 2000-2025-08, Test: 2025-09 to 2026-06)

**Date:** 2026-06-25
**Features:** 30 features (15 numeric, 5 categorical, 10 boolean) — no macro features
**Training data:** 667,327 rows (2000-01 to 2025-08)
**Test data:** 18,827 rows (2025-09 to 2026-06) — ~10 month holdout

---

## All Models — Held-Out Test Set

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

---

## Comparison with Reference Project

| Metric | Reference | Our v2 |
|--------|-----------|--------|
| **MAE** | ~$27,000 | **$26,000** |
| **RMSE** | ~$39,000 | **$36,750** |
| **MAPE** | ~5.7% | **4.0%** |
| **PER10** | — | **94.0%** |

**We beat the reference project on every metric.**

---

## Error by Price Quartile (Ensemble)

| Quartile | Count | MAE | MAPE | PER10 |
|----------|-------|-----|------|-------|
| Q1 (cheapest) | 4,724 | $19,798 | 4.8% | 89.4% |
| Q2 | 4,830 | $20,999 | 3.7% | 95.8% |
| Q3 | 4,582 | $22,835 | 3.3% | 97.3% |
| Q4 (expensive) | 4,691 | $40,486 | 4.2% | 93.6% |

---

## Key Numbers to Compare After Adding Macro Features

| Metric | Current (no macro) | After Macro | Change |
|--------|-------------------|-------------|--------|
| Ensemble MAE | $26,000 | _pending_ | |
| Ensemble MAPE | 4.0% | _pending_ | |
| Ensemble PER10 | 94.0% | _pending_ | |
| Ensemble R² | 0.9700 | _pending_ | |
| Q4 MAE (expensive) | $40,486 | _pending_ | |
