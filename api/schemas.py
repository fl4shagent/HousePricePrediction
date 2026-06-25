from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    town: str = Field(..., example="ANG MO KIO")
    flat_type: str = Field(..., example="4 ROOM")
    floor_area_sqm: float = Field(..., example=93.0)
    flat_model: str = Field(..., example="New Generation")
    storey_range: str = Field(..., example="07 TO 09")
    remaining_lease: str = Field(..., example="61 years 04 months")
    block: str = Field(..., example="406")
    street_name: str = Field(..., example="ANG MO KIO AVE 10")
    transaction_month: str = Field(..., example="2025-10")


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    model_version: str
    features_used: int
