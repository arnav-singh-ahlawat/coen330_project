"""Data loading helpers for the MetroPT-3 dataset."""

import pandas as pd

from src.config import RAW_DATA_FILE
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
