"""Evaluate the saved MetroPT-3 binary classifier."""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    CONFUSION_MATRIX_PLOT,
    METRICS_FILE,
    MODEL_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loading import load_raw_data, standardize_columns
from src.features import build_features
from src.labeling import add_target
from src.utils import ensure_directories, require_file


def evaluate():
    """Evaluate the saved model on the held-out split."""
    ensure_directories([METRICS_FILE.parent, CONFUSION_MATRIX_PLOT.parent])
    require_file(MODEL_FILE, "Saved model not found. Run python -m src.train first.")

    raw = standardize_columns(load_raw_data())
    labeled = add_target(raw)
    X, y = build_features(labeled)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_FILE)
    predictions = model.predict(X_test)

    metrics = pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        ]
    )
    metrics.to_csv(METRICS_FILE, index=False)

    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["normal", "failure_risk"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("MetroPT-3 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT, dpi=150)
    plt.close()

    print(classification_report(y_test, predictions, target_names=["normal", "failure_risk"]))
    print(metrics.to_string(index=False))
    print(f"Saved metrics to {METRICS_FILE}")
    print(f"Saved plot to {CONFUSION_MATRIX_PLOT}")
    return metrics


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    evaluate()
