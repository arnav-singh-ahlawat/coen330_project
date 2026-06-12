"""Train, tune thresholds, and select MetroPT-3 failure-risk classifiers."""

import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CONFUSION_MATRIX_PLOT,
    METRICS_FILE,
    MODEL_FILE,
    PRECISION_RECALL_CURVE_PLOT,
    RANDOM_STATE,
    TEST_METRICS_FILE,
    TEST_PROBABILITY_PLOT,
    THRESHOLD_TABLE_FILE,
    TRAINING_OUTPUT_FILE,
    VALIDATION_PROBABILITY_PLOT,
    WINDOWED_LABELED_DATA_FILE,
)
from src.modeling import (  # noqa: E402
    calculate_metrics,
    class_balance_text,
    describe_split_info,
    event_aware_blocked_split,
    format_params,
    load_windowed_dataset,
    metrics_at_threshold,
    predict_scores,
    probability_summary,
    split_features_labels,
)
from src.utils import ensure_directories  # noqa: E402


THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
FINAL_SELECTION_STRATEGY = "validation_f1_threshold_tuned"


def build_models():
    """Create the model set for failure-risk prediction comparison."""
    return {
        "Logistic Regression baseline": {
            "estimator": Pipeline(
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
            "param_grid": None,
        },
        "Decision Tree": {
            "estimator": Pipeline(
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
            "param_grid": {
                "classifier__max_depth": [4, 8, None],
                "classifier__min_samples_leaf": [1, 10],
            },
        },
        "Random Forest": {
            "estimator": Pipeline(
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
            "param_grid": {
                "classifier__max_depth": [8, None],
                "classifier__min_samples_leaf": [1, 10],
            },
        },
        "Extra Trees": {
            "estimator": Pipeline(
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
            "param_grid": None,
        },
        "HistGradientBoostingClassifier": {
            "estimator": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
                    ),
                ]
            ),
            "param_grid": {
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_leaf_nodes": [15, 31],
            },
        },
    }


def fit_model(name, spec, X_train, y_train):
    """Fit one model, tuning with a small grid when configured."""
    estimator = spec["estimator"]
    param_grid = spec["param_grid"]
    if not param_grid:
        estimator.fit(X_train, y_train)
        return estimator, {}

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"{name} best CV params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def tune_threshold(y_true, scores, metric_name):
    """Select the threshold that maximizes a validation metric."""
    rows = []
    for threshold in THRESHOLDS:
        _, metrics = metrics_at_threshold(y_true, scores, threshold)
        rows.append({"threshold": threshold, **metrics})
    threshold_metrics = pd.DataFrame(rows)
    best_threshold = float(
        threshold_metrics.sort_values(
            by=metric_name,
            ascending=False,
            kind="mergesort",
        ).iloc[0]["threshold"]
    )
    _, best_metrics = metrics_at_threshold(y_true, scores, best_threshold)
    return best_threshold, {"threshold": best_threshold, **best_metrics}


def select_best(table, metric_name):
    """Select a model row by one validation metric using stable model order."""
    return table.sort_values(
        by=metric_name,
        ascending=False,
        kind="mergesort",
    ).iloc[0]


def prefixed(prefix, values):
    """Prefix metric keys for flat result tables."""
    return {f"{prefix}_{key}": value for key, value in values.items()}


def print_confusion_matrix(label, metrics):
    """Print a compact binary confusion matrix from metric counts."""
    print(f"{label} confusion matrix [[tn, fp], [fn, tp]]:")
    print(f"  [[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]")


def plot_probability_distribution(y_true, scores, threshold, output_path, title):
    """Save a class-conditional probability distribution plot."""
    plot_data = pd.DataFrame(
        {
            "target": y_true.map({0: "normal", 1: "failure_risk"}).to_numpy(),
            "probability": scores,
        }
    )
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=plot_data,
        x="probability",
        hue="target",
        bins=50,
        stat="count",
        common_norm=False,
        element="step",
    )
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    plt.title(title)
    plt.xlabel("Predicted probability for failure-risk class")
    plt.ylabel("Window count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_precision_recall(y_true, scores, threshold, output_path, title):
    """Save a validation precision-recall curve when probabilities are available."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label="Precision-recall curve")
    if len(thresholds):
        threshold_index = int(np.argmin(np.abs(thresholds - threshold)))
        plt.scatter(
            recall[threshold_index],
            precision[threshold_index],
            color="red",
            label=f"selected threshold={threshold:.2f}",
        )
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train():
    """Train models, tune thresholds on validation only, and test event 4 once."""
    ensure_directories(
        [
            MODEL_FILE.parent,
            METRICS_FILE.parent,
            TEST_METRICS_FILE.parent,
            THRESHOLD_TABLE_FILE.parent,
            TRAINING_OUTPUT_FILE.parent,
            CONFUSION_MATRIX_PLOT.parent,
        ]
    )

    data = load_windowed_dataset(WINDOWED_LABELED_DATA_FILE)
    train_frame, validation_frame, test_frame, split_info = event_aware_blocked_split(data)
    X_train, y_train = split_features_labels(train_frame)
    X_validation, y_validation = split_features_labels(validation_frame)
    X_test, y_test = split_features_labels(test_frame)

    print(f"Processed dataset: {WINDOWED_LABELED_DATA_FILE}")
    print(f"Processed dataset shape: {data.shape}")
    print(f"Train shape: {train_frame.shape}")
    print(f"Validation shape: {validation_frame.shape}")
    print(f"Test shape: {test_frame.shape}")
    for line in describe_split_info(split_info):
        print(line)
    print(class_balance_text("Train class balance", y_train))
    print(class_balance_text("Validation class balance", y_validation))
    print(class_balance_text("Test class balance", y_test))
    print("window_start is used only for sorting/splitting and is not a model feature.")
    print("Threshold tuning uses validation probabilities only. Test data is not used for selection.")

    models = build_models()
    metrics_rows = []
    threshold_rows = []
    fitted_models = {}
    validation_scores_by_model = {}

    print("\nValidation metrics and threshold diagnostics:")
    for name, spec in models.items():
        print(f"\nTraining {name}")
        model, best_params = fit_model(name, spec, X_train, y_train)
        fitted_models[name] = model

        validation_scores = predict_scores(model, X_validation)
        if validation_scores is None:
            raise ValueError(f"{name} does not provide probabilities or decision scores.")
        validation_scores_by_model[name] = validation_scores
        validation_probability = probability_summary(validation_scores)

        default_predictions, default_metrics = metrics_at_threshold(
            y_validation,
            validation_scores,
            0.5,
        )
        best_f1_threshold, best_f1_metrics = tune_threshold(y_validation, validation_scores, "f1")
        best_f2_threshold, best_f2_metrics = tune_threshold(y_validation, validation_scores, "f2")

        metrics_rows.append(
            {
                "model": name,
                "best_params": format_params(best_params),
                "validation_probability_min": validation_probability["probability_min"],
                "validation_probability_mean": validation_probability["probability_mean"],
                "validation_probability_max": validation_probability["probability_max"],
                **prefixed("default_0_50", default_metrics),
                "best_f1_threshold": best_f1_threshold,
                **prefixed("best_f1_threshold", best_f1_metrics),
                "best_f2_threshold": best_f2_threshold,
                **prefixed("best_f2_threshold", best_f2_metrics),
            }
        )
        threshold_rows.append(
            {
                "model": name,
                "best_params": format_params(best_params),
                "default_threshold": 0.5,
                "default_validation_f1": default_metrics["f1"],
                "default_validation_f2": default_metrics["f2"],
                "best_f1_threshold": best_f1_threshold,
                "best_f1_validation_f1": best_f1_metrics["f1"],
                "best_f1_validation_f2": best_f1_metrics["f2"],
                "best_f2_threshold": best_f2_threshold,
                "best_f2_validation_f1": best_f2_metrics["f1"],
                "best_f2_validation_f2": best_f2_metrics["f2"],
                "validation_probability_min": validation_probability["probability_min"],
                "validation_probability_mean": validation_probability["probability_mean"],
                "validation_probability_max": validation_probability["probability_max"],
            }
        )

        print(
            f"{name} validation probability min/mean/max: "
            f"{validation_probability['probability_min']:.6f}/"
            f"{validation_probability['probability_mean']:.6f}/"
            f"{validation_probability['probability_max']:.6f}"
        )
        print(
            f"{name} default threshold 0.50: "
            f"precision={default_metrics['precision']:.4f}, "
            f"recall={default_metrics['recall']:.4f}, "
            f"f1={default_metrics['f1']:.4f}, f2={default_metrics['f2']:.4f}, "
            f"predicted_normal={default_metrics['predicted_normal_count']}, "
            f"predicted_failure={default_metrics['predicted_failure_count']}"
        )
        print_confusion_matrix(f"{name} validation default 0.50", default_metrics)
        print(
            f"{name} best validation F1 threshold {best_f1_threshold:.2f}: "
            f"precision={best_f1_metrics['precision']:.4f}, "
            f"recall={best_f1_metrics['recall']:.4f}, "
            f"f1={best_f1_metrics['f1']:.4f}, f2={best_f1_metrics['f2']:.4f}, "
            f"predicted_normal={best_f1_metrics['predicted_normal_count']}, "
            f"predicted_failure={best_f1_metrics['predicted_failure_count']}"
        )
        print_confusion_matrix(f"{name} validation best-F1 threshold", best_f1_metrics)
        print(
            f"{name} best validation F2 threshold {best_f2_threshold:.2f}: "
            f"precision={best_f2_metrics['precision']:.4f}, "
            f"recall={best_f2_metrics['recall']:.4f}, "
            f"f1={best_f2_metrics['f1']:.4f}, f2={best_f2_metrics['f2']:.4f}, "
            f"predicted_normal={best_f2_metrics['predicted_normal_count']}, "
            f"predicted_failure={best_f2_metrics['predicted_failure_count']}"
        )
        print_confusion_matrix(f"{name} validation best-F2 threshold", best_f2_metrics)

    metrics_table = pd.DataFrame(metrics_rows)
    metrics_table.to_csv(METRICS_FILE, index=False)

    threshold_table = pd.DataFrame(threshold_rows)
    default_winner = select_best(threshold_table, "default_validation_f1")
    tuned_f1_winner = select_best(threshold_table, "best_f1_validation_f1")
    tuned_f2_winner = select_best(threshold_table, "best_f2_validation_f2")
    selected_name = tuned_f1_winner["model"]
    selected_threshold = float(tuned_f1_winner["best_f1_threshold"])

    threshold_table["selected_by_default_f1"] = threshold_table["model"] == default_winner["model"]
    threshold_table["selected_by_tuned_f1"] = threshold_table["model"] == tuned_f1_winner["model"]
    threshold_table["selected_by_tuned_f2"] = threshold_table["model"] == tuned_f2_winner["model"]
    threshold_table["selected_final"] = threshold_table["model"] == selected_name
    threshold_table["final_selection_strategy"] = np.where(
        threshold_table["selected_final"],
        FINAL_SELECTION_STRATEGY,
        "",
    )
    threshold_table.to_csv(THRESHOLD_TABLE_FILE, index=False)

    final_model = fitted_models[selected_name]
    joblib.dump(final_model, MODEL_FILE)

    test_scores = predict_scores(final_model, X_test)
    test_probability = probability_summary(test_scores)
    test_predictions, test_metrics = metrics_at_threshold(y_test, test_scores, selected_threshold)
    test_metrics_table = pd.DataFrame(
        [
            {
                "model": selected_name,
                "selection_strategy": FINAL_SELECTION_STRATEGY,
                "threshold": selected_threshold,
                **test_metrics,
                "test_probability_min": test_probability["probability_min"],
                "test_probability_mean": test_probability["probability_mean"],
                "test_probability_max": test_probability["probability_max"],
            }
        ]
    )
    test_metrics_table.to_csv(TEST_METRICS_FILE, index=False)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_predictions,
        labels=[0, 1],
        display_labels=["normal", "failure_risk"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title("MetroPT-3 Failure-Risk Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT, dpi=150)
    plt.close()

    selected_validation_scores = validation_scores_by_model[selected_name]
    plot_probability_distribution(
        y_validation,
        selected_validation_scores,
        selected_threshold,
        VALIDATION_PROBABILITY_PLOT,
        f"Validation Probability Distribution - {selected_name}",
    )
    plot_probability_distribution(
        y_test,
        test_scores,
        selected_threshold,
        TEST_PROBABILITY_PLOT,
        f"Test Probability Distribution - {selected_name}",
    )
    plot_precision_recall(
        y_validation,
        selected_validation_scores,
        selected_threshold,
        PRECISION_RECALL_CURVE_PLOT,
        f"Validation Precision-Recall Curve - {selected_name}",
    )

    print("\nModel selection strategy comparison:")
    print(
        f"Best default-threshold validation F1: {default_winner['model']} "
        f"(threshold=0.50, f1={default_winner['default_validation_f1']:.4f})"
    )
    print(
        f"Best tuned-threshold validation F1: {tuned_f1_winner['model']} "
        f"(threshold={float(tuned_f1_winner['best_f1_threshold']):.2f}, "
        f"f1={tuned_f1_winner['best_f1_validation_f1']:.4f})"
    )
    print(
        f"Best tuned-threshold validation F2: {tuned_f2_winner['model']} "
        f"(threshold={float(tuned_f2_winner['best_f2_threshold']):.2f}, "
        f"f2={tuned_f2_winner['best_f2_validation_f2']:.4f})"
    )
    print(
        f"\nSelected final model: {selected_name} using {FINAL_SELECTION_STRATEGY} "
        f"at threshold {selected_threshold:.2f}"
    )
    print("Final held-out event-4 test metrics; accuracy is secondary:")
    print(test_metrics_table.to_string(index=False))
    print(
        f"Final test probability min/mean/max: "
        f"{test_probability['probability_min']:.6f}/"
        f"{test_probability['probability_mean']:.6f}/"
        f"{test_probability['probability_max']:.6f}"
    )
    print_confusion_matrix("Final event-4 test", test_metrics)
    print(f"Saved validation metrics table to {METRICS_FILE}")
    print(f"Saved threshold table to {THRESHOLD_TABLE_FILE}")
    print(f"Saved final model to {MODEL_FILE}")
    print(f"Saved final test metrics to {TEST_METRICS_FILE}")
    print(f"Saved final test confusion matrix plot to {CONFUSION_MATRIX_PLOT}")
    print(f"Saved validation probability plot to {VALIDATION_PROBABILITY_PLOT}")
    print(f"Saved test probability plot to {TEST_PROBABILITY_PLOT}")
    print(f"Saved validation precision-recall curve to {PRECISION_RECALL_CURVE_PLOT}")
    return final_model


def main():
    """Run training and save console output to results/training_output.txt."""
    sns.set_theme(style="whitegrid")
    ensure_directories([TRAINING_OUTPUT_FILE.parent])
    with TRAINING_OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        with redirect_stdout(output_file):
            train()
    print(TRAINING_OUTPUT_FILE.read_text(encoding="utf-8"), end="")
    print(f"Saved training output to {TRAINING_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
