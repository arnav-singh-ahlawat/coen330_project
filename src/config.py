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
FAILURE_RISK_WINDOWS = []

DROP_COLUMNS = [TARGET_COLUMN]
