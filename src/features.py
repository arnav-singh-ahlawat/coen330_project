"""Feature engineering for MetroPT-3 classification."""

import pandas as pd

from src.config import TARGET_COLUMN, TIMESTAMP_COLUMN


def create_minute_window_features(df, timestamp_column=TIMESTAMP_COLUMN, window="1min"):
    """Aggregate raw sensor records into fixed-width timestamp windows."""
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' is missing.")

    windowed = df.copy()
    windowed[timestamp_column] = pd.to_datetime(windowed[timestamp_column], errors="coerce")
    invalid_count = windowed[timestamp_column].isna().sum()
    if invalid_count == len(windowed):
        raise ValueError(f"Column '{timestamp_column}' could not be parsed as datetimes.")

    windowed = windowed.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
    numeric = windowed.select_dtypes(include=["number", "bool"]).copy()
    drop_columns = [
        column
        for column in numeric.columns
        if column == TARGET_COLUMN or column.lower().startswith("unnamed")
    ]
    numeric = numeric.drop(columns=drop_columns, errors="ignore")
    if numeric.empty:
        raise ValueError("No numeric sensor columns were found for window feature creation.")

    numeric[timestamp_column] = windowed[timestamp_column]
    aggregates = (
        numeric.set_index(timestamp_column)
        .resample(window)
        .agg(["mean", "std", "min", "max", "last"])
        .dropna(how="all")
    )
    aggregates.columns = [
        f"{column}_{stat}" for column, stat in aggregates.columns.to_flat_index()
    ]
    aggregates = aggregates.reset_index().rename(columns={timestamp_column: "window_start"})
    aggregates["row_count"] = (
        windowed.set_index(timestamp_column)
        .resample(window)
        .size()
        .reindex(aggregates["window_start"])
        .to_numpy()
    )
    return aggregates


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
