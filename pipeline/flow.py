"""Prefect pipeline — orchestrates the full HDB price prediction workflow.

Converts the notebook-based pipeline into a production-ready orchestrated flow
with task dependencies, retries, caching, and data quality validation.

Run: python pipeline/flow.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from prefect import flow, task

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.config import (
    ENABLE_MACRO_FEATURES,
    GEOCODING_BATCH_SIZE,
    GEOCODING_CONCURRENCY,
    INTERIM_DIR,
    MODEL_VERSION,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SHAPEFILE_PATH,
    SPLIT_MONTH,
    SPLIT_YEAR,
)
from src.data_collection import (
    MATURE_ESTATES,
    fetch_all_resale_transactions,
    fetch_cpi,
    fetch_datagov_csv,
    get_mrt_stations,
    get_sora_3m,
)
from src.feature_engineering import (
    classify_elite_school,
    compute_cbd_distance,
    compute_nearest_from_locations,
    compute_nearest_mrt,
    compute_nearest_school_by_level,
    geocode_buildings,
    geocode_schools,
)
from src.preprocessing import (
    convert_yn_to_bool,
    extract_transaction_date,
    parse_remaining_lease,
    parse_storey_range,
)

# ─── DATA INGESTION ──────────────────────────────────────────────────────────

@task(retries=3, retry_delay_seconds=30, log_prints=True)
def fetch_transactions():
    return fetch_all_resale_transactions(RAW_DIR)


@task(log_prints=True)
def fetch_mrt_stations():
    df = get_mrt_stations(SHAPEFILE_PATH)
    df.to_csv(f"{RAW_DIR}/mrt_lrt_stations.csv", index=False)
    print(f"MRT stations: {len(df)}")
    return df


@task(retries=3, retry_delay_seconds=30, log_prints=True)
def fetch_schools():
    path = f"{RAW_DIR}/schools.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return fetch_datagov_csv("d_688b934f82c1059ed0a6993d2a829089", path)


@task(retries=3, retry_delay_seconds=30, log_prints=True)
def fetch_property_info():
    path = f"{RAW_DIR}/hdb_property_info.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return fetch_datagov_csv("d_17f5382f26140b1fdae0ba2ef6239d2f", path)


@task(log_prints=True)
def fetch_macro_data():
    sora = get_sora_3m()
    sora.to_csv(f"{RAW_DIR}/sora_3m.csv", index=False)
    cpi = fetch_cpi(f"{RAW_DIR}/cpi.csv")
    print(f"SORA: {len(sora)} months, CPI: {len(cpi)} months")
    return sora, cpi


# ─── PREPROCESSING ───────────────────────────────────────────────────────────

@task(log_prints=True)
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicates")

    df = extract_transaction_date(df)
    df["remaining_lease_months"] = parse_remaining_lease(
        df["remaining_lease"], df["month"], df["lease_commence_date"]
    )
    df["storey_median"] = parse_storey_range(df["storey_range"])
    df = df.drop(columns=["remaining_lease", "storey_range", "lease_commence_date"])

    print(f"Cleaned: {len(df)} rows, {len(df.columns)} columns")
    return df


# ─── GEOSPATIAL FEATURES ────────────────────────────────────────────────────

@task(retries=2, retry_delay_seconds=60, log_prints=True)
def geocode_all_buildings(df: pd.DataFrame) -> pd.DataFrame:
    unique = df[["block", "street_name"]].drop_duplicates().reset_index(drop=True)
    return geocode_buildings(
        unique, cache_path=f"{INTERIM_DIR}/building_geocodes.csv",
        concurrency=GEOCODING_CONCURRENCY, batch_size=GEOCODING_BATCH_SIZE,
    )


@task(retries=2, retry_delay_seconds=60, log_prints=True)
def geocode_all_schools(schools_raw: pd.DataFrame) -> pd.DataFrame:
    return geocode_schools(schools_raw, cache_path=f"{INTERIM_DIR}/schools_geocoded.csv")


@task(log_prints=True)
def compute_mrt_distances(df: pd.DataFrame, mrt_df: pd.DataFrame) -> pd.DataFrame:
    cache = f"{INTERIM_DIR}/mrt_distances.csv"
    df["month_dt"] = pd.to_datetime(df["month"])

    if os.path.exists(cache):
        cached = pd.read_csv(cache)
        cached["month_dt"] = pd.to_datetime(cached["month_dt"])
        print(f"Loaded {len(cached)} cached MRT distances")
        df = df.merge(
            cached[["block", "street_name", "month_dt", "dist_to_nearest_mrt_km", "closest_mrt"]],
            on=["block", "street_name", "month_dt"], how="left"
        )
        return df

    building_month = df[["block", "street_name", "latitude", "longitude", "month_dt"]].drop_duplicates(
        subset=["block", "street_name", "month_dt"]
    ).reset_index(drop=True)

    print(f"Computing MRT distances for {len(building_month)} pairs...")
    results = []
    for i, row in building_month.iterrows():
        if pd.isna(row["latitude"]):
            results.append((np.nan, None))
            continue
        dist, name = compute_nearest_mrt(row["latitude"], row["longitude"], mrt_df, row["month_dt"])
        results.append((dist, name))
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1}/{len(building_month)}")

    building_month["dist_to_nearest_mrt_km"] = [r[0] for r in results]
    building_month["closest_mrt"] = [r[1] for r in results]
    building_month[["block", "street_name", "month_dt", "dist_to_nearest_mrt_km", "closest_mrt"]].to_csv(cache, index=False)

    df = df.merge(
        building_month[["block", "street_name", "month_dt", "dist_to_nearest_mrt_km", "closest_mrt"]],
        on=["block", "street_name", "month_dt"], how="left"
    )
    return df


@task(log_prints=True)
def compute_school_flags(building_coords: pd.DataFrame, schools_geo: pd.DataFrame) -> pd.DataFrame:
    cache = f"{INTERIM_DIR}/school_flags.csv"
    if os.path.exists(cache):
        print("Loading cached school flags")
        return pd.read_csv(cache)

    buildings = building_coords[building_coords["latitude"].notna()].reset_index(drop=True)
    print(f"Computing school flags for {len(buildings)} buildings...")

    results = []
    for i, row in buildings.iterrows():
        pri, sec, mixed = compute_nearest_school_by_level(row["latitude"], row["longitude"], schools_geo)
        results.append({"block": row["block"], "street_name": row["street_name"],
                        "closest_pri_sch": pri, "closest_sec_sch": sec, "closest_mixed_sch": mixed})

    school_df = pd.DataFrame(results)
    for level in ["pri", "sec", "mixed"]:
        school_df[f"is_elite_closest_{level}_sch"] = school_df[f"closest_{level}_sch"].apply(
            lambda x: classify_elite_school(x, schools_geo))

    flags = school_df[["block", "street_name", "is_elite_closest_pri_sch",
                       "is_elite_closest_sec_sch", "is_elite_closest_mixed_sch"]]
    flags.to_csv(cache, index=False)
    return flags


@task(log_prints=True)
def compute_spatial_features(building_coords: pd.DataFrame) -> pd.DataFrame:
    cache = f"{INTERIM_DIR}/spatial_distances.csv"
    if os.path.exists(cache):
        print("Loading cached spatial distances")
        return pd.read_csv(cache)

    malls = pd.read_csv(f"{RAW_DIR}/shopping_malls.csv")
    hawker = pd.read_csv(f"{RAW_DIR}/hawker_centres.csv")
    buildings = building_coords[building_coords["latitude"].notna()].reset_index(drop=True)

    print(f"Computing spatial distances for {len(buildings)} buildings...")
    results = []
    for _, row in buildings.iterrows():
        lat, lng = row["latitude"], row["longitude"]
        mall_dist, _ = compute_nearest_from_locations(lat, lng, malls, "lat", "lng")
        hawker_dist, _ = compute_nearest_from_locations(lat, lng, hawker, "lat", "lng")
        results.append({"block": row["block"], "street_name": row["street_name"],
                        "dist_to_cbd_km": compute_cbd_distance(lat, lng),
                        "dist_to_nearest_mall_km": mall_dist,
                        "dist_to_nearest_hawker_km": hawker_dist})

    spatial_df = pd.DataFrame(results)
    spatial_df.to_csv(cache, index=False)
    return spatial_df


@task(log_prints=True)
def compute_carpark_features(building_coords: pd.DataFrame) -> pd.DataFrame:
    cache = f"{INTERIM_DIR}/carpark_features.csv"
    if os.path.exists(cache):
        print("Loading cached carpark features")
        return pd.read_csv(cache)

    from pyproj import Transformer
    carpark = pd.read_csv(f"{RAW_DIR}/hdb_carpark_info.csv")
    transformer = Transformer.from_crs("EPSG:3414", "EPSG:4326", always_xy=True)
    carpark["x_coord"] = pd.to_numeric(carpark["x_coord"], errors="coerce")
    carpark["y_coord"] = pd.to_numeric(carpark["y_coord"], errors="coerce")
    valid = carpark["x_coord"].notna() & carpark["y_coord"].notna()
    lngs, lats = transformer.transform(carpark.loc[valid, "x_coord"].values, carpark.loc[valid, "y_coord"].values)
    carpark.loc[valid, "lat"] = lats
    carpark.loc[valid, "lng"] = lngs

    buildings = building_coords[building_coords["latitude"].notna()].reset_index(drop=True)
    print(f"Matching carparks to {len(buildings)} buildings...")

    results = []
    for _, row in buildings.iterrows():
        dist, idx = compute_nearest_from_locations(row["latitude"], row["longitude"], carpark[carpark["lat"].notna()], "lat", "lng")
        cp_type = carpark.loc[idx, "car_park_type"] if idx is not None else None
        results.append({"block": row["block"], "street_name": row["street_name"],
                        "nearest_carpark_dist_km": dist, "nearest_carpark_type": cp_type})

    carpark_df = pd.DataFrame(results)
    carpark_df.to_csv(cache, index=False)
    return carpark_df


# ─── ASSEMBLY & TRAINING ────────────────────────────────────────────────────

@task(log_prints=True)
def assemble_features(df, building_coords, school_flags, spatial_df, carpark_df, property_df, sora_cpi):
    df = df.merge(building_coords, on=["block", "street_name"], how="left")
    df = df.merge(school_flags, on=["block", "street_name"], how="left")
    df = df.merge(spatial_df, on=["block", "street_name"], how="left")
    df = df.merge(carpark_df, on=["block", "street_name"], how="left")

    if ENABLE_MACRO_FEATURES and sora_cpi is not None:
        sora_df, cpi_df = sora_cpi
        sora_df["lag_year"] = sora_df["year"]
        sora_df["lag_month"] = sora_df["month"] + 1
        mask = sora_df["lag_month"] > 12
        sora_df.loc[mask, "lag_month"] = 1
        sora_df.loc[mask, "lag_year"] += 1

        cpi_df["lag_year"] = cpi_df["year"]
        cpi_df["lag_month"] = cpi_df["month"] + 1
        mask = cpi_df["lag_month"] > 12
        cpi_df.loc[mask, "lag_month"] = 1
        cpi_df.loc[mask, "lag_year"] += 1

        df = df.merge(sora_df[["lag_year", "lag_month", "sora_3m"]].rename(
            columns={"lag_year": "transaction_year", "lag_month": "transaction_month", "sora_3m": "sora_3m_lagged"}),
            on=["transaction_year", "transaction_month"], how="left")
        df = df.merge(cpi_df[["lag_year", "lag_month", "cpi"]].rename(
            columns={"lag_year": "transaction_year", "lag_month": "transaction_month", "cpi": "cpi_lagged"}),
            on=["transaction_year", "transaction_month"], how="left")
        df["sora_3m_lagged"] = df["sora_3m_lagged"].bfill().ffill()
        df["cpi_lagged"] = df["cpi_lagged"].ffill().bfill()

    # Property info
    prop = property_df.copy()
    time_varying = [c for c in prop.columns if "sold" in c or "rental" in c]
    prop = prop.drop(columns=time_varying)
    prop = prop.rename(columns={"blk_no": "block", "street": "street_name"})
    yn_cols = ["residential", "commercial", "market_hawker", "miscellaneous", "multistorey_carpark", "precinct_pavilion"]
    prop = convert_yn_to_bool(prop, yn_cols)
    df = df.merge(prop, on=["block", "street_name"], how="left")

    df["is_mature_estate"] = df["town"].str.upper().isin(MATURE_ESTATES)
    df = df.dropna(subset=["latitude", "max_floor_lvl"])
    df = df.drop(columns=["block", "street_name", "month", "month_dt", "bldg_contract_town"], errors="ignore")

    print(f"Assembled: {len(df)} rows, {len(df.columns)} columns")
    return df


@task(log_prints=True)
def split_and_save(df):
    train_mask = (df["transaction_year"] < SPLIT_YEAR) | (
        (df["transaction_year"] == SPLIT_YEAR) & (df["transaction_month"] < SPLIT_MONTH))
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[~train_mask].reset_index(drop=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_train.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    df_test.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")
    return df_train, df_test


@task(log_prints=True)
def validate_data(df_train, df_test):
    from tests.data_quality.validate import validate_processed_training
    result = validate_processed_training(df_train, df_test)
    status = "PASSED" if result["success"] else "FAILED"
    print(f"Data quality: {status} — {result['passed']}/{result['total']} expectations")
    if not result["success"]:
        raise ValueError(f"Data quality validation failed: {result['failed']} expectations failed")
    return result


@task(log_prints=True)
def train_models(df_train):
    from lightgbm import LGBMRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBRegressor

    TARGET = "resale_price"
    y = df_train[TARGET]
    X = df_train.drop(columns=[TARGET])

    cat_cols = ["town", "flat_type", "flat_model", "closest_mrt", "nearest_carpark_type"]
    bool_cols = [c for c in X.columns if X[c].dtype == "bool"]
    num_cols = [c for c in X.columns if c not in cat_cols + bool_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("bool", "passthrough", bool_cols),
    ])

    X_t = preprocessor.fit_transform(X)

    lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.1, num_leaves=63,
                         colsample_bytree=0.8, reg_lambda=0.5, min_child_samples=20,
                         n_jobs=-1, random_state=42, verbose=-1)
    lgbm.fit(X_t, y)

    xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=8,
                       colsample_bytree=0.8, reg_lambda=0.5, gamma=0.3,
                       n_jobs=-1, random_state=42, verbosity=0)
    xgb.fit(X_t, y)

    version_dir = f"{MODELS_DIR}/{MODEL_VERSION}"
    os.makedirs(version_dir, exist_ok=True)
    joblib.dump(lgbm, f"{version_dir}/lgbm_model.joblib")
    joblib.dump(xgb, f"{version_dir}/xgb_model.joblib")
    joblib.dump(preprocessor, f"{version_dir}/preprocessor.joblib")

    joblib.dump(lgbm, f"{MODELS_DIR}/lgbm_model.joblib")
    joblib.dump(xgb, f"{MODELS_DIR}/xgb_model.joblib")
    joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.joblib")

    config = {"numeric_cols": num_cols, "categorical_cols": cat_cols,
              "boolean_cols": bool_cols, "target": TARGET, "version": MODEL_VERSION}
    with open(f"{MODELS_DIR}/feature_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Models saved to {version_dir}/")
    return lgbm, xgb, preprocessor


# ─── MAIN FLOW ───────────────────────────────────────────────────────────────

@flow(name="HDB Price Prediction Pipeline", log_prints=True)
def hdb_pipeline():
    """Full pipeline: fetch → clean → engineer → validate → train."""
    print("Starting HDB Price Prediction Pipeline")

    # 1. Data ingestion (parallel where possible)
    df_raw = fetch_transactions()
    mrt_df = fetch_mrt_stations()
    schools_raw = fetch_schools()
    property_df = fetch_property_info()
    sora_cpi = fetch_macro_data() if ENABLE_MACRO_FEATURES else None

    # 2. Preprocessing
    df = clean_transactions(df_raw)

    # 3. Geospatial features
    building_coords = geocode_all_buildings(df)
    df = df.merge(building_coords[["block", "street_name", "latitude", "longitude"]],
                  on=["block", "street_name"], how="left")
    df = compute_mrt_distances(df, mrt_df)

    schools_geo = geocode_all_schools(schools_raw)
    school_flags = compute_school_flags(building_coords, schools_geo)
    spatial_df = compute_spatial_features(building_coords)
    carpark_df = compute_carpark_features(building_coords)

    # 4. Assembly
    df = df.drop(columns=["latitude", "longitude"], errors="ignore")
    df = assemble_features(df, building_coords, school_flags, spatial_df, carpark_df, property_df, sora_cpi)

    # 5. Split and validate
    df_train, df_test = split_and_save(df)
    validate_data(df_train, df_test)

    # 6. Train
    train_models(df_train)

    print("Pipeline complete!")


if __name__ == "__main__":
    hdb_pipeline()
