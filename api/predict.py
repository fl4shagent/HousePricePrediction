import joblib
import json
import pandas as pd
import numpy as np
import os
import re

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "interim")

_lgbm = None
_xgb = None
_preprocessor = None
_feature_config = None
_building_lookup = None

MATURE_ESTATES = {
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CENTRAL AREA", "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA",
    "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON",
    "TAMPINES", "TOA PAYOH",
}


def load_models():
    global _lgbm, _xgb, _preprocessor, _feature_config, _building_lookup

    _lgbm = joblib.load(os.path.join(MODELS_DIR, "lgbm_model.joblib"))
    _xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    _preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))

    with open(os.path.join(MODELS_DIR, "feature_config.json")) as f:
        _feature_config = json.load(f)

    geo = pd.read_csv(os.path.join(INTERIM_DIR, "building_geocodes.csv"), dtype={"block": str})
    spatial = pd.read_csv(os.path.join(INTERIM_DIR, "spatial_distances.csv"), dtype={"block": str})
    schools = pd.read_csv(os.path.join(INTERIM_DIR, "school_flags.csv"), dtype={"block": str})
    carpark = pd.read_csv(os.path.join(INTERIM_DIR, "carpark_features.csv"), dtype={"block": str})

    lookup = geo.merge(spatial, on=["block", "street_name"], how="left")
    lookup = lookup.merge(schools, on=["block", "street_name"], how="left")
    lookup = lookup.merge(carpark, on=["block", "street_name"], how="left")
    _building_lookup = lookup.set_index(["block", "street_name"])


def _parse_remaining_lease(lease_str: str) -> float:
    years, months = 0, 0
    y_match = re.search(r"(\d+)\s*year", lease_str)
    m_match = re.search(r"(\d+)\s*month", lease_str)
    if y_match:
        years = int(y_match.group(1))
    if m_match:
        months = int(m_match.group(1))
    return years * 12 + months


def _parse_storey_range(storey_str: str) -> float:
    parts = storey_str.split(" TO ")
    return (int(parts[0]) + int(parts[1])) / 2


def predict_price(request) -> dict:
    year = int(request.transaction_month[:4])
    month = int(request.transaction_month[5:7])

    remaining_months = _parse_remaining_lease(request.remaining_lease)
    storey_median = _parse_storey_range(request.storey_range)

    key = (str(request.block), request.street_name)
    if key in _building_lookup.index:
        bldg = _building_lookup.loc[key]
        if isinstance(bldg, pd.DataFrame):
            bldg = bldg.iloc[0]
        lat = bldg.get("latitude", np.nan)
        lng = bldg.get("longitude", np.nan)
        dist_mrt = bldg.get("dist_to_nearest_mrt_km", np.nan) if "dist_to_nearest_mrt_km" in bldg.index else np.nan
        dist_cbd = bldg.get("dist_to_cbd_km", np.nan)
        dist_mall = bldg.get("dist_to_nearest_mall_km", np.nan)
        dist_hawker = bldg.get("dist_to_nearest_hawker_km", np.nan)
        carpark_dist = bldg.get("nearest_carpark_dist_km", np.nan)
        carpark_type = bldg.get("nearest_carpark_type", "MULTI-STOREY CAR PARK")
        elite_pri = bldg.get("is_elite_closest_pri_sch", False)
        elite_sec = bldg.get("is_elite_closest_sec_sch", False)
        elite_mixed = bldg.get("is_elite_closest_mixed_sch", False)
        closest_mrt = bldg.get("closest_mrt", "Unknown") if "closest_mrt" in bldg.index else "Unknown"
    else:
        lat, lng = np.nan, np.nan
        dist_mrt = dist_cbd = dist_mall = dist_hawker = carpark_dist = np.nan
        carpark_type = "MULTI-STOREY CAR PARK"
        elite_pri = elite_sec = elite_mixed = False
        closest_mrt = "Unknown"

    # Use median values for missing geospatial features
    if pd.isna(lat):
        lat, lng = 1.36, 103.84
        dist_mrt, dist_cbd = 0.65, 10.0
        dist_mall, dist_hawker = 0.8, 0.5
        carpark_dist = 0.1

    is_mature = request.town.upper() in MATURE_ESTATES

    sora_3m = 2.0
    cpi = 100.0

    row = pd.DataFrame([{
        "town": request.town,
        "flat_type": request.flat_type,
        "floor_area_sqm": request.floor_area_sqm,
        "flat_model": request.flat_model,
        "transaction_year": year,
        "transaction_month": month,
        "remaining_lease_months": remaining_months,
        "storey_median": storey_median,
        "latitude": lat,
        "longitude": lng,
        "dist_to_nearest_mrt_km": dist_mrt,
        "closest_mrt": closest_mrt,
        "dist_to_cbd_km": dist_cbd,
        "dist_to_nearest_mall_km": dist_mall,
        "dist_to_nearest_hawker_km": dist_hawker,
        "nearest_carpark_dist_km": carpark_dist,
        "nearest_carpark_type": carpark_type if pd.notna(carpark_type) else "MULTI-STOREY CAR PARK",
        "sora_3m_lagged": sora_3m,
        "cpi_lagged": cpi,
        "max_floor_lvl": 12.0,
        "year_completed": 1990.0,
        "total_dwelling_units": 120.0,
        "is_elite_closest_pri_sch": bool(elite_pri),
        "is_elite_closest_sec_sch": bool(elite_sec),
        "is_elite_closest_mixed_sch": bool(elite_mixed),
        "residential": True,
        "commercial": False,
        "market_hawker": False,
        "miscellaneous": False,
        "multistorey_carpark": False,
        "precinct_pavilion": False,
        "is_mature_estate": is_mature,
    }])

    X = _preprocessor.transform(row)
    X = pd.DataFrame(X)
    pred_lgbm = _lgbm.predict(X)[0]
    pred_xgb = _xgb.predict(X)[0]
    pred_ensemble = (pred_lgbm + pred_xgb) / 2

    return {
        "predicted_price": round(float(pred_ensemble), 2),
        "predicted_price_formatted": f"${pred_ensemble:,.0f}",
        "model_version": _feature_config.get("version", "unknown"),
        "features_used": X.shape[1],
    }
