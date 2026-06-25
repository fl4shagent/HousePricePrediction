"""Pipeline configuration — all tuneable parameters in one place."""

RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

SHAPEFILE_PATH = "TrainStation_Mar2026/RapidTransitSystemStation.shp"

SPLIT_YEAR = 2025
SPLIT_MONTH = 9

ENABLE_MACRO_FEATURES = True

GEOCODING_CONCURRENCY = 10
GEOCODING_BATCH_SIZE = 500

MODEL_VERSION = "v3_macro"
