"""Project configuration for the MetroPT-3 classification workflow."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RAW_DATA_FILE = RAW_DATA_DIR / "MetroPT3(AirCompressor).csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "metropt3_labeled_features.csv"
WINDOWED_LABELED_DATA_FILE = PROCESSED_DATA_DIR / "windowed_labeled_data.csv"
MODEL_FILE = MODELS_DIR / "metropt3_binary_classifier.joblib"
METRICS_FILE = RESULTS_DIR / "metrics_table.csv"
CONFUSION_MATRIX_PLOT = PLOTS_DIR / "confusion_matrix.png"

TARGET_COLUMN = "target"
TIMESTAMP_COLUMN = "timestamp"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# If the raw data does not already contain a binary target column, configure
# timestamp windows here after documenting the project labeling assumptions.
# Format: ("YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD HH:MM:SS")
#
# Failure-risk starts one hour before each known failure start and continues
# through the failure end time.
FAILURE_RISK_WINDOWS = [
    ("2020-04-17 23:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 22:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 09:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 13:30:00", "2020-07-15 19:00:00"),
]

DROP_COLUMNS = [TARGET_COLUMN]
