"""Run an optimistic stratified window-level baseline experiment.

This script intentionally does not replace the event-aware train/validation/test
pipeline. It mixes processed windows across train and test using a stratified
random split, so it is useful only as a feature-separability comparison.
"""

import inspect
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    PLOTS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    WINDOWED_LABELED_DATA_FILE,
)
from src.modeling import (  # noqa: E402
    calculate_metrics,
    class_balance_text,
    load_windowed_dataset,
    predict_scores,
    split_features_labels,
)
from src.utils import ensure_directories  # noqa: E402


STRATIFIED_BASELINE_METRICS_FILE = RESULTS_DIR / "stratified_baseline_metrics.csv"
STRATIFIED_BASELINE_CONFUSION_MATRIX_PLOT = (
    PLOTS_DIR / "stratified_baseline_confusion_matrix.png"
)
TEST_SIZE = 0.20
BASELINE_NOTE = (
    "Optimistic window-level comparison only: this stratified split mixes windows "
    "from the same failure events across train/test and can overestimate "
    "deployment performance."
)


def hist_gradient_boosting_classifier():
    """Build HistGradientBoostingClassifier with balanced classes when available."""
    params = {"random_state": RANDOM_STATE}
    if "class_weight" in inspect.signature(HistGradientBoostingClassifier).parameters:
        params["class_weight"] = "balanced"
    return HistGradientBoostingClassifier(**params)


def build_models():
    """Create the five model types used in the project comparison."""
    return {
        "Logistic Regression baseline": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "HistGradientBoostingClassifier": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", hist_gradient_boosting_classifier()),
            ]
        ),
    }


def class_counts(y):
    """Return class counts for a binary target series."""
    counts = y.value_counts().sort_index()
    return {
        "class_0": int(counts.get(0, 0)),
        "class_1": int(counts.get(1, 0)),
    }


def plot_confusion_matrices(rows, output_path):
    """Save all model confusion matrices in one comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for index, row in enumerate(rows):
        matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["normal", "failure_risk"],
            yticklabels=["normal", "failure_risk"],
            ax=axes[index],
        )
        axes[index].set_title(row["model"])
        axes[index].set_xlabel("Predicted")
        axes[index].set_ylabel("Actual")

    for axis in axes[len(rows) :]:
        axis.axis("off")

    fig.suptitle("Stratified Window-Level Baseline Confusion Matrices", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_stratified_baseline():
    """Train and evaluate the optimistic stratified window-level baseline."""
    ensure_directories(
        [
            STRATIFIED_BASELINE_METRICS_FILE.parent,
            STRATIFIED_BASELINE_CONFUSION_MATRIX_PLOT.parent,
        ]
    )

    data = load_windowed_dataset(WINDOWED_LABELED_DATA_FILE)
    X, y = split_features_labels(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_counts = class_counts(y_train)
    test_counts = class_counts(y_test)

    print(f"Processed dataset: {WINDOWED_LABELED_DATA_FILE}")
    print(f"Processed dataset shape: {data.shape}")
    print(f"Feature count after dropping target/window_start: {X.shape[1]}")
    print("Split strategy: stratified random window-level train/test split")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(class_balance_text("Train class balance", y_train))
    print(class_balance_text("Test class balance", y_test))
    print("window_start is used only by preprocessing and is not a model feature.")
    print(BASELINE_NOTE)

    rows = []
    for name, model in build_models().items():
        print(f"\nTraining {name}")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores = predict_scores(model, X_test)
        metrics = calculate_metrics(y_test, predictions, scores)
        row = {
            "model": name,
            "split_strategy": "stratified random window-level train/test split",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_count": X.shape[1],
            "train_class_0": train_counts["class_0"],
            "train_class_1": train_counts["class_1"],
            "test_class_0": test_counts["class_0"],
            "test_class_1": test_counts["class_1"],
            **metrics,
            "interpretation_note": BASELINE_NOTE,
        }
        rows.append(row)
        print(
            f"{name}: accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"f2={metrics['f2']:.4f}, roc_auc={metrics['roc_auc']:.4f}"
        )
        print(
            f"{name} confusion matrix [[tn, fp], [fn, tp]]: "
            f"[[{metrics['tn']}, {metrics['fp']}], "
            f"[{metrics['fn']}, {metrics['tp']}]]"
        )

    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(STRATIFIED_BASELINE_METRICS_FILE, index=False)
    plot_confusion_matrices(rows, STRATIFIED_BASELINE_CONFUSION_MATRIX_PLOT)

    print(f"\nSaved stratified baseline metrics to {STRATIFIED_BASELINE_METRICS_FILE}")
    print(
        "Saved stratified baseline confusion matrix plot to "
        f"{STRATIFIED_BASELINE_CONFUSION_MATRIX_PLOT}"
    )
    return metrics_table


def main():
    """Run the stratified baseline experiment."""
    sns.set_theme(style="whitegrid")
    run_stratified_baseline()


if __name__ == "__main__":
    main()
