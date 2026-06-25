from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.predict import load_models, predict_price
from api.schemas import PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="HDB Resale Price Predictor",
    description="Predicts Singapore HDB resale flat prices using LGBM + XGBoost ensemble",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    result = predict_price(request)
    return PredictionResponse(**result)
