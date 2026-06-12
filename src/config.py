"""Project configuration for the MetroPT-3 failure-risk prediction workflow."""

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
MODEL_FILE = MODELS_DIR / "final_model.joblib"
METRICS_FILE = RESULTS_DIR / "metrics_table.csv"
TEST_METRICS_FILE = RESULTS_DIR / "test_metrics.csv"
THRESHOLD_TABLE_FILE = RESULTS_DIR / "threshold_table.csv"
TRAINING_OUTPUT_FILE = RESULTS_DIR / "training_output.txt"
CONFUSION_MATRIX_PLOT = PLOTS_DIR / "confusion_matrix.png"
VALIDATION_PROBABILITY_PLOT = PLOTS_DIR / "validation_probability_distribution.png"
TEST_PROBABILITY_PLOT = PLOTS_DIR / "test_probability_distribution.png"
PRECISION_RECALL_CURVE_PLOT = PLOTS_DIR / "precision_recall_curve.png"

TARGET_COLUMN = "target"
TIMESTAMP_COLUMN = "timestamp"

RANDOM_STATE = 42

# Configure timestamp windows for the binary failure-risk target.
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
