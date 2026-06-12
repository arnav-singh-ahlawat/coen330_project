"""Demo entrypoint for the MetroPT-3 failure-risk classifier."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from src.config import MODEL_FILE
from src.data_loading import load_raw_data, standardize_columns
from src.features import build_features
from src.labeling import add_target
from src.utils import require_file


def main(sample_size=5):
    """Load the saved model and print predictions for a small local sample."""
    require_file(MODEL_FILE, "Saved model not found. Run python -m src.train first.")

    raw = standardize_columns(load_raw_data())
    labeled = add_target(raw)
    X, _ = build_features(labeled)
    sample = X.head(sample_size)

    model = joblib.load(MODEL_FILE)
    predictions = model.predict(sample)

    for index, prediction in zip(sample.index, predictions):
        label = "failure-risk/anomaly" if int(prediction) == 1 else "normal"
        print(f"row={index} prediction={label}")


if __name__ == "__main__":
    main()
