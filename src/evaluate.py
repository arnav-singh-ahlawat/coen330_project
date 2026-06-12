"""Evaluate the saved MetroPT-3 failure-risk classifier on event 4."""

import os
import sys
from pathlib import Path

import joblib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CONFUSION_MATRIX_PLOT,
    MODEL_FILE,
    TEST_METRICS_FILE,
    THRESHOLD_TABLE_FILE,
    WINDOWED_LABELED_DATA_FILE,
)
from src.modeling import (  # noqa: E402
    class_balance_text,
    describe_split_info,
    event_aware_blocked_split,
    load_windowed_dataset,
    metrics_at_threshold,
    predict_scores,
    probability_summary,
    split_features_labels,
)
from src.utils import ensure_directories, require_file  # noqa: E402


def evaluate():
    """Evaluate the saved final model on the held-out failure-event test split."""
    ensure_directories([TEST_METRICS_FILE.parent, CONFUSION_MATRIX_PLOT.parent])
    require_file(MODEL_FILE, "Saved final model not found. Run python -m src.train first.")
    require_file(
        THRESHOLD_TABLE_FILE,
        "Threshold table not found. Run python -m src.train before evaluation.",
    )

    data = load_windowed_dataset(WINDOWED_LABELED_DATA_FILE)
    train_frame, validation_frame, test_frame, split_info = event_aware_blocked_split(data)
    X_test, y_test = split_features_labels(test_frame)

    print(f"Processed dataset: {WINDOWED_LABELED_DATA_FILE}")
    print(f"Processed dataset shape: {data.shape}")
    print(f"Train shape: {train_frame.shape}")
    print(f"Validation shape: {validation_frame.shape}")
    print(f"Test shape: {test_frame.shape}")
    for line in describe_split_info(split_info):
        print(line)
    print(class_balance_text("Test class balance", y_test))
    print("window_start is used only for sorting/splitting and is not a model feature.")

    model = joblib.load(MODEL_FILE)
    scores = predict_scores(model, X_test)
    threshold_table = pd.read_csv(THRESHOLD_TABLE_FILE)
    selected_rows = threshold_table[threshold_table["selected_final"] == True]
    if selected_rows.empty:
        raise ValueError("Threshold table does not identify a selected final model.")
    selected_row = selected_rows.iloc[0]
    selected_model = selected_row["model"]
    selected_threshold = float(selected_row["best_f1_threshold"])

    predictions, metrics = metrics_at_threshold(y_test, scores, selected_threshold)
    probability_stats = probability_summary(scores)

    metrics_table = pd.DataFrame(
        [
            {
                "model": selected_model,
                "selection_strategy": selected_row["final_selection_strategy"],
                "threshold": selected_threshold,
                **metrics,
                "test_probability_min": probability_stats["probability_min"],
                "test_probability_mean": probability_stats["probability_mean"],
                "test_probability_max": probability_stats["probability_max"],
            }
        ]
    )
    metrics_table.to_csv(TEST_METRICS_FILE, index=False)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["normal", "failure_risk"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("MetroPT-3 Failure-Risk Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT, dpi=150)
    plt.close()

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["normal", "failure_risk"],
            zero_division=0,
        )
    )
    print("Final held-out event-4 test metrics; accuracy is secondary:")
    print(metrics_table.to_string(index=False))
    print(
        f"Final threshold selected on validation: {selected_threshold:.2f}; "
        f"predicted_normal={metrics['predicted_normal_count']}, "
        f"predicted_failure={metrics['predicted_failure_count']}"
    )
    print(
        "Test probability min/mean/max: "
        f"{probability_stats['probability_min']:.6f}/"
        f"{probability_stats['probability_mean']:.6f}/"
        f"{probability_stats['probability_max']:.6f}"
    )
    print(f"Saved final test metrics to {TEST_METRICS_FILE}")
    print(f"Saved plot to {CONFUSION_MATRIX_PLOT}")
    return metrics_table


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    evaluate()
