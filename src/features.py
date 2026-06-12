"""Feature engineering for MetroPT-3 classification."""

import pandas as pd

from src.config import TARGET_COLUMN, TIMESTAMP_COLUMN


def add_time_features(df, timestamp_column=TIMESTAMP_COLUMN):
    """Add simple calendar features when a timestamp column is available."""
    if timestamp_column not in df.columns:
        return df

    featured = df.copy()
    timestamps = pd.to_datetime(featured[timestamp_column], errors="coerce")
    if timestamps.isna().all():
        return featured.drop(columns=[timestamp_column])

    featured["hour"] = timestamps.dt.hour
    featured["dayofweek"] = timestamps.dt.dayofweek
    featured["month"] = timestamps.dt.month
    return featured.drop(columns=[timestamp_column])


def build_features(df, target_column=TARGET_COLUMN):
    """Prepare feature matrix X and target vector y."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing.")

    featured = add_time_features(df)
    y = featured[target_column].astype(int)
    X = featured.drop(columns=[target_column])
    X = X.select_dtypes(include=["number", "bool"]).copy()

    if X.empty:
        raise ValueError("No numeric feature columns were found after preprocessing.")

    return X, y


def save_processed_features(X, y, path):
    """Save processed numeric features and target for inspection."""
    output = X.copy()
    output[TARGET_COLUMN] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
