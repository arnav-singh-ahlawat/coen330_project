"""Shared modeling helpers for the MetroPT-3 failure-risk pipeline."""

import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    FAILURE_RISK_WINDOWS,
    TARGET_COLUMN,
    WINDOWED_LABELED_DATA_FILE,
)
from src.utils import require_file


WINDOW_START_COLUMN = "window_start"


def load_windowed_dataset(path=WINDOWED_LABELED_DATA_FILE):
    """Load the processed 1-minute window dataset used by the ML pipeline."""
    require_file(path, "Processed windowed data not found. Run python -m src.preprocessing first.")
    data = pd.read_csv(path)
    missing = {WINDOW_START_COLUMN, TARGET_COLUMN}.difference(data.columns)
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {sorted(missing)}")

    data[WINDOW_START_COLUMN] = pd.to_datetime(data[WINDOW_START_COLUMN], errors="coerce")
    if data[WINDOW_START_COLUMN].isna().any():
        bad_rows = int(data[WINDOW_START_COLUMN].isna().sum())
        raise ValueError(f"Processed dataset has {bad_rows} invalid window_start values.")

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    return data.sort_values(WINDOW_START_COLUMN).reset_index(drop=True)


def parse_failure_events(failure_windows=None):
    """Return configured failure-risk windows as ordered event records."""
    windows = FAILURE_RISK_WINDOWS if failure_windows is None else failure_windows
    if len(windows) < 3:
        raise ValueError("At least three failure events are required for train/validation/test blocking.")

    events = []
    for event_id, (start, end) in enumerate(windows, start=1):
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)
        if pd.isna(start_time) or pd.isna(end_time) or end_time < start_time:
            raise ValueError(f"Invalid failure event window: {(start, end)}")
        events.append({"event_id": event_id, "start": start_time, "end": end_time})

    events = sorted(events, key=lambda event: event["start"])
    for previous, current in zip(events, events[1:]):
        if current["start"] <= previous["end"]:
            raise ValueError("Failure event windows must not overlap.")
    return events


def assign_failure_events(data, failure_windows=None):
    """Assign target-positive windows to their configured failure event."""
    events = parse_failure_events(failure_windows)
    event_ids = pd.Series(pd.NA, index=data.index, dtype="Int64")
    timestamps = data[WINDOW_START_COLUMN]

    for event in events:
        mask = (timestamps >= event["start"]) & (timestamps <= event["end"])
        event_ids.loc[mask] = event["event_id"]

    unassigned_positive_count = int(((data[TARGET_COLUMN] == 1) & event_ids.isna()).sum())
    if unassigned_positive_count:
        raise ValueError(
            f"{unassigned_positive_count} positive target rows are outside configured failure events."
        )
    return event_ids


def _midpoint(left, right):
    """Return the midpoint between two timestamps."""
    return left + (right - left) / 2


def _event_block_boundaries(data, events):
    """Create non-overlapping time blocks around ordered failure events."""
    data_start = data[WINDOW_START_COLUMN].min()
    data_end = data[WINDOW_START_COLUMN].max() + pd.Timedelta(nanoseconds=1)
    boundaries = []

    for index, event in enumerate(events):
        if index == 0:
            block_start = data_start
        else:
            block_start = _midpoint(events[index - 1]["end"], event["start"])

        if index == len(events) - 1:
            block_end = data_end
        else:
            block_end = _midpoint(event["end"], events[index + 1]["start"])

        boundaries.append(
            {
                "event_id": event["event_id"],
                "event_start": event["start"],
                "event_end": event["end"],
                "block_start": block_start,
                "block_end": block_end,
            }
        )
    return boundaries


def _validate_split(name, frame):
    """Ensure each split has both normal and failure-risk windows."""
    counts = frame[TARGET_COLUMN].value_counts()
    if int(counts.get(0, 0)) == 0 or int(counts.get(1, 0)) == 0:
        raise ValueError(f"{name} split must contain both normal and failure-risk windows.")


def event_aware_blocked_split(data, failure_windows=None):
    """Split processed windows by failure-event blocks without random row sampling."""
    events = parse_failure_events(failure_windows)
    assign_failure_events(data, failure_windows)

    if len(events) == 4:
        train_events = (events[0]["event_id"], events[1]["event_id"])
        validation_events = (events[2]["event_id"],)
        test_events = (events[3]["event_id"],)
    else:
        train_events = tuple(event["event_id"] for event in events[:-2])
        validation_events = (events[-2]["event_id"],)
        test_events = (events[-1]["event_id"],)

    boundaries = _event_block_boundaries(data, events)
    split_event_ids = {
        "train": train_events,
        "validation": validation_events,
        "test": test_events,
    }

    split_frames = {}
    for split_name, event_ids in split_event_ids.items():
        mask = pd.Series(False, index=data.index)
        for boundary in boundaries:
            if boundary["event_id"] not in event_ids:
                continue
            block_mask = (
                (data[WINDOW_START_COLUMN] >= boundary["block_start"])
                & (data[WINDOW_START_COLUMN] < boundary["block_end"])
            )
            mask = mask | block_mask
        split_frames[split_name] = data.loc[mask].copy().sort_values(WINDOW_START_COLUMN)
        _validate_split(split_name, split_frames[split_name])

    split_info = {
        "strategy": "event-aware blocked split",
        "train_events": train_events,
        "validation_events": validation_events,
        "test_events": test_events,
        "blocks": boundaries,
    }
    return split_frames["train"], split_frames["validation"], split_frames["test"], split_info


def describe_split_info(split_info):
    """Return printable lines describing the event-aware split."""
    lines = [
        f"Split strategy: {split_info['strategy']}",
        f"Train failure events: {list(split_info['train_events'])}",
        f"Validation failure events: {list(split_info['validation_events'])}",
        f"Test failure events: {list(split_info['test_events'])}",
    ]
    for block in split_info["blocks"]:
        lines.append(
            "Event {event_id}: event={event_start} to {event_end}, "
            "assigned block={block_start} to {block_end}".format(**block)
        )
    return lines


def split_features_labels(frame):
    """Return numeric model features and labels, excluding window_start."""
    y = frame[TARGET_COLUMN].astype(int)
    X = frame.drop(columns=[TARGET_COLUMN, WINDOW_START_COLUMN])
    X = X.select_dtypes(include=["number", "bool"]).copy()
    if X.empty:
        raise ValueError("No numeric model features found after dropping target and window_start.")
    return X, y


def class_balance_text(name, y):
    """Build a concise class balance string for console output."""
    counts = y.value_counts().sort_index()
    total = len(y)
    pieces = []
    for class_value in [0, 1]:
        count = int(counts.get(class_value, 0))
        percent = (count / total) * 100 if total else 0
        pieces.append(f"class {class_value}: {count} ({percent:.2f}%)")
    return f"{name}: " + ", ".join(pieces)


def predict_scores(model, X):
    """Return positive-class scores when the estimator supports them."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def calculate_metrics(y_true, predictions, scores=None):
    """Calculate binary classification metrics for the failure-risk class."""
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    predicted_normal_count = int((predictions == 0).sum())
    predicted_failure_count = int((predictions == 1).sum())
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "f2": fbeta_score(y_true, predictions, beta=2, zero_division=0),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
        "predicted_normal_count": predicted_normal_count,
        "predicted_failure_count": predicted_failure_count,
    }
    if scores is not None and y_true.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
    else:
        metrics["roc_auc"] = None
    return metrics


def metrics_at_threshold(y_true, scores, threshold):
    """Calculate metrics after converting positive-class scores at a threshold."""
    predictions = (scores >= threshold).astype(int)
    return predictions, calculate_metrics(y_true, predictions, scores)


def probability_summary(scores):
    """Return min/mean/max for positive-class probabilities."""
    return {
        "probability_min": float(scores.min()),
        "probability_mean": float(scores.mean()),
        "probability_max": float(scores.max()),
    }


def format_params(params):
    """Serialize model parameters for result tables."""
    if not params:
        return ""
    return json.dumps(params, sort_keys=True)
