"""Data loading helpers for the MetroPT-3 dataset."""

import pandas as pd

from src.config import RAW_DATA_FILE, TIMESTAMP_COLUMN
from src.utils import require_file


def load_raw_data(path=RAW_DATA_FILE):
    """Load the local raw MetroPT-3 CSV file."""
    path = require_file(
        path,
        "Raw dataset not found. Download MetroPT-3 and place the CSV in data/raw/.",
    )
    return pd.read_csv(path)


def standardize_columns(df):
    """Normalize column names for consistent downstream processing."""
    clean = df.copy()
    clean.columns = (
        clean.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return clean


def detect_timestamp_column(df, preferred_column=TIMESTAMP_COLUMN):
    """Find the timestamp column using names first, then parseable values."""
    if preferred_column in df.columns:
        parsed = pd.to_datetime(df[preferred_column], errors="coerce")
        if parsed.notna().any():
            return preferred_column

    name_candidates = [
        column
        for column in df.columns
        if any(token in column.lower() for token in ("timestamp", "datetime", "date", "time"))
    ]
    for column in name_candidates:
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().any():
            return column

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        sample = df[column].dropna().head(1000)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return column

    raise ValueError(
        "Timestamp column could not be found. Expected a parseable column named "
        f"'{preferred_column}' or another date/time-like column."
    )
