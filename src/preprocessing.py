"""Preprocessing and windowed dataset generation for failure-risk prediction."""

import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FAILURE_RISK_WINDOWS,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    WINDOWED_LABELED_DATA_FILE,
)
from src.data_loading import detect_timestamp_column, load_raw_data, standardize_columns
from src.features import create_minute_window_features
from src.labeling import label_from_failure_windows


def build_preprocessor():
    """Build a numeric preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _print_class_balance(df):
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = len(df)
    print("Class balance:")
    for class_value, count in counts.items():
        percent = (count / total) * 100 if total else 0
        print(f"  {class_value}: {count} ({percent:.2f}%)")


def run_preprocessing(
    raw_path=RAW_DATA_FILE,
    output_path=WINDOWED_LABELED_DATA_FILE,
    failure_risk_windows=None,
):
    """Create one-minute labeled feature windows from the raw MetroPT-3 CSV."""
    windows = FAILURE_RISK_WINDOWS if failure_risk_windows is None else failure_risk_windows
    if not windows:
        raise ValueError(
            "Failure windows are missing. Configure FAILURE_RISK_WINDOWS in "
            "src/config.py before creating windowed labels."
        )

    print(f"Loading raw data from {raw_path}")
    raw = load_raw_data(raw_path)
    print(f"Raw data shape: {raw.shape}")

    raw = standardize_columns(raw)
    timestamp_column = detect_timestamp_column(raw)
    print(f"Detected timestamp column: {timestamp_column}")
    raw[timestamp_column] = pd.to_datetime(raw[timestamp_column], errors="coerce")

    print("Creating 1-minute window features")
    windowed = create_minute_window_features(raw, timestamp_column=timestamp_column)
    print(f"Number of 1-minute windows: {len(windowed)}")

    print("Adding binary failure-risk target from configured failure windows")
    labeled = label_from_failure_windows(
        windowed,
        timestamp_column="window_start",
        target_column=TARGET_COLUMN,
        failure_risk_windows=windows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)
    print(f"Output path: {output_path}")
    print(f"Final shape: {labeled.shape}")
    _print_class_balance(labeled)
    return labeled


def main():
    """Run preprocessing as a command-line script."""
    try:
        run_preprocessing()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
