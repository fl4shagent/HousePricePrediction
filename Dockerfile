FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY api/ api/
COPY src/ src/
COPY models/lgbm_model.joblib models/
COPY models/xgb_model.joblib models/
COPY models/preprocessor.joblib models/
COPY models/feature_config.json models/
COPY data/interim/ data/interim/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
