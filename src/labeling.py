"""Label creation for supervised failure-risk binary classification."""

import pandas as pd

from src.config import FAILURE_RISK_WINDOWS, TARGET_COLUMN, TIMESTAMP_COLUMN


def validate_binary_target(df, target_column=TARGET_COLUMN):
    """Validate that the target column is binary."""
    values = set(df[target_column].dropna().unique())
    if not values.issubset({0, 1, False, True}):
        raise ValueError(
            f"{target_column} must be binary with values 0/1. Found: {sorted(values)}"
        )
    labeled = df.copy()
    labeled[target_column] = labeled[target_column].astype(int)
    return labeled


def label_from_failure_windows(
    df,
    timestamp_column=TIMESTAMP_COLUMN,
    target_column=TARGET_COLUMN,
    failure_risk_windows=None,
):
    """Create a binary early-warning target from failure-risk timestamp windows."""
    windows = FAILURE_RISK_WINDOWS if failure_risk_windows is None else failure_risk_windows
    if not windows:
        raise ValueError(
            "No target column found and no FAILURE_RISK_WINDOWS configured. "
            "Set documented labeling windows in src/config.py before training."
        )
    if timestamp_column not in df.columns:
        raise ValueError(
            f"Cannot create labels because timestamp column '{timestamp_column}' is missing."
        )

    labeled = df.copy()
    timestamps = pd.to_datetime(labeled[timestamp_column], errors="coerce")
    if timestamps.isna().all():
        raise ValueError(f"Column '{timestamp_column}' could not be parsed as datetimes.")

    target = pd.Series(0, index=labeled.index, dtype=int)
    for start, end in windows:
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)
        target.loc[(timestamps >= start_time) & (timestamps <= end_time)] = 1

    labeled[target_column] = target
    return labeled


def add_target(df, target_column=TARGET_COLUMN):
    """Return a dataframe with a validated binary target column."""
    if target_column in df.columns:
        return validate_binary_target(df, target_column)
    return label_from_failure_windows(df, target_column=target_column)
