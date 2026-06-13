"""Demo entrypoint for the MetroPT-3 failure-risk classifier."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.config import MODEL_FILE, THRESHOLD_TABLE_FILE, WINDOWED_LABELED_DATA_FILE
from src.modeling import WINDOW_START_COLUMN, load_windowed_dataset, split_features_labels
from src.utils import require_file


CLASS_NAMES = {
    0: "normal",
    1: "failure-risk",
}

DEMO_NOTE = (
    "This demo illustrates the pipeline usage on processed 1-minute windows. "
    "The current model is not deployment-ready because event-aware testing showed "
    "poor generalization to held-out failure event 4."
)


def load_final_threshold(path=THRESHOLD_TABLE_FILE):
    """Load the selected final threshold, defaulting only when the table is absent."""
    if not path.exists():
        print(f"WARNING: {path} is missing. Defaulting to threshold 0.50.")
        return 0.5, "default threshold because threshold_table.csv is missing"

    table = pd.read_csv(path)
    required_columns = {"selected_final", "final_selection_strategy"}
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required threshold selection columns: {sorted(missing_columns)}"
        )

    selected_mask = table["selected_final"].astype(str).str.lower().isin({"true", "1"})
    selected_rows = table.loc[selected_mask]
    if len(selected_rows) != 1:
        raise ValueError(f"{path} must contain exactly one selected_final row.")

    selected = selected_rows.iloc[0]
    strategy = str(selected["final_selection_strategy"])
    threshold_columns_by_strategy = {
        "validation_f1_threshold_tuned": "best_f1_threshold",
        "validation_f2_threshold_tuned": "best_f2_threshold",
        "default_threshold": "default_threshold",
    }
    threshold_column = threshold_columns_by_strategy.get(strategy)
    if threshold_column is None:
        raise ValueError(f"Unsupported final threshold selection strategy: {strategy}")
    if threshold_column not in selected.index:
        raise ValueError(f"{path} is missing threshold column: {threshold_column}")

    return float(selected[threshold_column]), f"{path} ({threshold_column})"


def validate_feature_names(model, X):
    """Fail early with a clear feature-schema error before calling sklearn predict."""
    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is None:
        return

    expected_features = list(expected_features)
    actual_features = list(X.columns)
    if actual_features == expected_features:
        return

    missing = sorted(set(expected_features).difference(actual_features))
    extra = sorted(set(actual_features).difference(expected_features))
    raise ValueError(
        "Processed demo features do not match the saved model feature schema. "
        f"Missing features: {missing}. Extra features: {extra}."
    )


def select_demo_indices(y):
    """Select one normal row and one failure-risk row when available."""
    indices = []

    normal_indices = y[y == 0].head(1).index.tolist()
    failure_risk_indices = y[y == 1].head(1).index.tolist()

    if normal_indices:
        indices.append(normal_indices[0])
    if failure_risk_indices:
        indices.append(failure_risk_indices[0])

    if not indices:
        raise ValueError("Processed dataset does not contain any demo samples.")
    return indices


def predict_failure_probabilities(model, X):
    """Return predicted probabilities for class 1 using predict_proba."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Saved model does not support predict_proba.")

    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        failure_risk_index = classes.index(1)
    elif probabilities.shape[1] > 1:
        failure_risk_index = 1
    else:
        raise ValueError("Saved model predict_proba output does not include class 1.")

    return probabilities[:, failure_risk_index]


def main():
    """Load processed windows and print predictions for representative samples."""
    require_file(
        WINDOWED_LABELED_DATA_FILE,
        "Processed windowed data not found. Run python -m src.preprocessing first.",
    )
    require_file(MODEL_FILE, "Saved model not found. Run python -m src.train first.")

    data = load_windowed_dataset(WINDOWED_LABELED_DATA_FILE)
    X, y = split_features_labels(data)
    selected_indices = select_demo_indices(y)
    sample_features = X.loc[selected_indices]
    sample_labels = y.loc[selected_indices]

    model = joblib.load(MODEL_FILE)
    validate_feature_names(model, X)

    threshold, threshold_source = load_final_threshold(THRESHOLD_TABLE_FILE)
    probabilities = predict_failure_probabilities(model, sample_features)
    predictions = (probabilities >= threshold).astype(int)

    print("MetroPT-3 failure-risk demo")
    print(f"Processed data: {WINDOWED_LABELED_DATA_FILE}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Feature columns used: {len(X.columns)}")
    print(f"Threshold used: {threshold:.2f}")
    print(f"Threshold source: {threshold_source}")
    print()

    for row_index, probability, prediction in zip(selected_indices, probabilities, predictions):
        true_class = int(sample_labels.loc[row_index])
        window_start = data.loc[row_index, WINDOW_START_COLUMN]
        print(f"sample type: {CLASS_NAMES[true_class]}")
        print(f"window_start: {window_start}")
        print(f"predicted probability of failure-risk: {probability:.6f}")
        print(f"threshold used: {threshold:.2f}")
        print(f"predicted class: {CLASS_NAMES[int(prediction)]} ({int(prediction)})")
        print(f"true class: {CLASS_NAMES[true_class]} ({true_class})")
        print()

    if not (sample_labels == 1).any():
        print("WARNING: No failure-risk sample was available in the processed dataset.")

    print(f"Note: {DEMO_NOTE}")


if __name__ == "__main__":
    main()
