"""Data quality validation using Great Expectations v1.

Three validation suites:
1. Raw transactions — after data fetch
2. Geocoded buildings — after geocoding
3. Processed training data — before model training

Run: python tests/data_quality/validate.py
"""

import os

import great_expectations as gx
import pandas as pd
from great_expectations.expectations import (
    ExpectColumnMeanToBeBetween,
    ExpectColumnToExist,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToNotBeNull,
    ExpectTableRowCountToBeBetween,
)


def _validate(df: pd.DataFrame, suite_name: str, expectations: list) -> dict:
    """Run a list of expectations against a dataframe."""
    context = gx.get_context()

    data_source = context.data_sources.add_or_update_pandas(name=suite_name)
    data_asset = data_source.add_dataframe_asset(name=f"{suite_name}_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(f"{suite_name}_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name=suite_name, expectations=expectations)

    results = batch.validate(suite)

    stats = results.statistics
    return {
        "suite": suite_name,
        "success": results.success,
        "total": stats["evaluated_expectations"],
        "passed": stats["successful_expectations"],
        "failed": stats["unsuccessful_expectations"],
        "rate": f"{stats['success_percent']:.1f}%",
    }


def validate_raw_transactions(df: pd.DataFrame) -> dict:
    """Validate raw resale transaction data after fetch."""
    return _validate(df, "raw_transactions", [
        ExpectColumnToExist(column="month"),
        ExpectColumnToExist(column="town"),
        ExpectColumnToExist(column="flat_type"),
        ExpectColumnToExist(column="resale_price"),
        ExpectColumnToExist(column="floor_area_sqm"),

        ExpectColumnValuesToNotBeNull(column="month"),
        ExpectColumnValuesToNotBeNull(column="town"),
        ExpectColumnValuesToNotBeNull(column="flat_type"),
        ExpectColumnValuesToNotBeNull(column="resale_price"),
        ExpectColumnValuesToNotBeNull(column="floor_area_sqm"),
        ExpectColumnValuesToNotBeNull(column="block"),
        ExpectColumnValuesToNotBeNull(column="street_name"),

        ExpectColumnValuesToBeBetween(column="resale_price", min_value=10000, max_value=3000000),
        ExpectColumnValuesToBeBetween(column="floor_area_sqm", min_value=20, max_value=400),

        ExpectColumnValuesToBeInSet(
            column="flat_type",
            value_set=["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM",
                       "EXECUTIVE", "MULTI-GENERATION"]
        ),

        ExpectTableRowCountToBeBetween(min_value=500000, max_value=1000000),
    ])


def validate_geocoded_buildings(df: pd.DataFrame) -> dict:
    """Validate geocoded building data."""
    result = _validate(df, "geocoded_buildings", [
        ExpectColumnToExist(column="block"),
        ExpectColumnToExist(column="street_name"),
        ExpectColumnToExist(column="latitude"),
        ExpectColumnToExist(column="longitude"),

        ExpectColumnValuesToNotBeNull(column="latitude", mostly=0.95),

        ExpectColumnValuesToBeBetween(column="latitude", min_value=1.2, max_value=1.5, mostly=0.95),
        ExpectColumnValuesToBeBetween(column="longitude", min_value=103.6, max_value=104.1, mostly=0.95),
    ])

    result["geocoding_success_rate"] = f"{df['latitude'].notna().mean():.1%}"
    return result


def validate_processed_training(df_train: pd.DataFrame, df_test: pd.DataFrame) -> dict:
    """Validate processed train/test data before model training."""
    result = _validate(df_train, "processed_training", [
        ExpectColumnValuesToNotBeNull(column="resale_price"),
        ExpectColumnValuesToNotBeNull(column="floor_area_sqm"),
        ExpectColumnValuesToNotBeNull(column="remaining_lease_months"),
        ExpectColumnValuesToNotBeNull(column="storey_median"),
        ExpectColumnValuesToNotBeNull(column="latitude"),
        ExpectColumnValuesToNotBeNull(column="dist_to_nearest_mrt_km"),
        ExpectColumnValuesToNotBeNull(column="dist_to_cbd_km"),

        ExpectColumnValuesToBeBetween(column="dist_to_nearest_mrt_km", min_value=0, max_value=10),
        ExpectColumnValuesToBeBetween(column="dist_to_cbd_km", min_value=0, max_value=30),
        ExpectColumnValuesToBeBetween(column="remaining_lease_months", min_value=1, max_value=1250),
        ExpectColumnValuesToBeBetween(column="storey_median", min_value=1, max_value=55),

        ExpectTableRowCountToBeBetween(min_value=500000, max_value=800000),

        ExpectColumnMeanToBeBetween(column="resale_price", min_value=200000, max_value=700000),
    ])

    result["train_rows"] = len(df_train)
    result["test_rows"] = len(df_test)
    result["train_nulls"] = int(df_train.isnull().sum().sum())
    result["test_nulls"] = int(df_test.isnull().sum().sum())
    return result


def run_all_validations():
    """Run all three validation suites on current data."""
    print("=" * 60)
    print("DATA QUALITY VALIDATION")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    all_passed = True

    # Suite 1: Raw transactions
    print("\n[1/3] Raw Transactions")
    raw_path = os.path.join(data_dir, "raw", "resale_transactions.csv")
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path, low_memory=False)
        r = validate_raw_transactions(df_raw)
        _print(r)
        if not r["success"]:
            all_passed = False
    else:
        print("  SKIP — file not found")

    # Suite 2: Geocoded buildings
    print("\n[2/3] Geocoded Buildings")
    geo_path = os.path.join(data_dir, "interim", "building_geocodes.csv")
    if os.path.exists(geo_path):
        df_geo = pd.read_csv(geo_path, dtype={"block": str})
        r = validate_geocoded_buildings(df_geo)
        _print(r)
        if not r["success"]:
            all_passed = False
    else:
        print("  SKIP — file not found")

    # Suite 3: Processed training data
    print("\n[3/3] Processed Training Data")
    train_path = os.path.join(data_dir, "processed", "train.csv")
    test_path = os.path.join(data_dir, "processed", "test.csv")
    if os.path.exists(train_path) and os.path.exists(test_path):
        df_train = pd.read_csv(train_path, low_memory=False)
        df_test = pd.read_csv(test_path, low_memory=False)
        r = validate_processed_training(df_train, df_test)
        _print(r)
        if not r["success"]:
            all_passed = False
    else:
        print("  SKIP — files not found")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL SUITES PASSED")
    else:
        print("SOME SUITES FAILED — check above for details")
    print("=" * 60)

    return all_passed


def _print(r: dict):
    status = "PASSED" if r["success"] else "FAILED"
    print(f"  {status} — {r['passed']}/{r['total']} expectations ({r['rate']})")
    for k, v in r.items():
        if k not in ("suite", "success", "total", "passed", "failed", "rate"):
            print(f"    {k}: {v}")


if __name__ == "__main__":
    success = run_all_validations()
    exit(0 if success else 1)
