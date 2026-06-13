"""Generate exploratory data analysis outputs for MetroPT-3 failure-risk data."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    FAILURE_RISK_WINDOWS,
    PLOTS_DIR,
    RESULTS_DIR,
    TARGET_COLUMN,
    WINDOWED_LABELED_DATA_FILE,
)
from src.modeling import (  # noqa: E402
    WINDOW_START_COLUMN,
    event_aware_blocked_split,
    load_windowed_dataset,
    parse_failure_events,
    split_features_labels,
)
from src.utils import ensure_directories  # noqa: E402


CLASS_BALANCE_PLOT = PLOTS_DIR / "class_balance.png"
SPLIT_CLASS_BALANCE_PLOT = PLOTS_DIR / "split_class_balance.png"
SENSOR_CORRELATION_HEATMAP = PLOTS_DIR / "sensor_correlation_heatmap.png"
FAILURE_WINDOWS_TIMELINE_PLOT = PLOTS_DIR / "failure_windows_timeline.png"
KEY_SENSOR_DISTRIBUTIONS_PLOT = PLOTS_DIR / "key_sensor_distributions.png"
EDA_SUMMARY_FILE = RESULTS_DIR / "eda_summary.csv"
EVENT_SUMMARY_FILE = RESULTS_DIR / "event_summary.csv"

CLASS_LABELS = {
    0: "normal",
    1: "failure-risk",
}
FAILURE_LEAD_TIME = pd.Timedelta(hours=1)
MAX_CORRELATION_FEATURES = 12
MAX_DISTRIBUTION_FEATURES = 6


def class_balance_frame(frame, split_name=None):
    """Return target counts and percentages for a dataframe."""
    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    total = len(frame)
    rows = []
    for class_value in [0, 1]:
        count = int(counts.get(class_value, 0))
        percent = (count / total) * 100 if total else 0.0
        row = {
            "class_value": class_value,
            "class_label": CLASS_LABELS[class_value],
            "count": count,
            "percentage": percent,
        }
        if split_name is not None:
            row["split"] = split_name
        rows.append(row)
    return pd.DataFrame(rows)


def add_bar_labels(axis):
    """Add integer labels to a bar chart when supported by the matplotlib version."""
    for container in axis.containers:
        axis.bar_label(container, fmt="%.0f", label_type="edge", padding=2)


def plot_class_balance(data):
    """Save the overall target class-balance plot."""
    plot_data = class_balance_frame(data)

    plt.figure(figsize=(7, 5))
    axis = sns.barplot(
        data=plot_data,
        x="class_label",
        y="count",
        hue="class_label",
        palette=["#4C78A8", "#E45756"],
        legend=False,
    )
    add_bar_labels(axis)
    axis.set_title("MetroPT-3 Target Class Balance")
    axis.set_xlabel("Target class")
    axis.set_ylabel("1-minute window count")
    plt.tight_layout()
    plt.savefig(CLASS_BALANCE_PLOT, dpi=150)
    plt.close()


def plot_split_class_balance(train_frame, validation_frame, test_frame):
    """Save target class balance for the event-aware train/validation/test split."""
    plot_data = pd.concat(
        [
            class_balance_frame(train_frame, "train"),
            class_balance_frame(validation_frame, "validation"),
            class_balance_frame(test_frame, "test"),
        ],
        ignore_index=True,
    )

    plt.figure(figsize=(9, 5))
    axis = sns.barplot(
        data=plot_data,
        x="split",
        y="count",
        hue="class_label",
        palette=["#4C78A8", "#E45756"],
    )
    add_bar_labels(axis)
    axis.set_title("Event-Aware Split Class Balance")
    axis.set_xlabel("Split")
    axis.set_ylabel("1-minute window count")
    axis.legend(title="Target class")
    plt.tight_layout()
    plt.savefig(SPLIT_CLASS_BALANCE_PLOT, dpi=150)
    plt.close()


def select_target_correlated_features(data, limit=MAX_CORRELATION_FEATURES):
    """Select engineered sensor features with the largest absolute target correlation."""
    numeric = data.select_dtypes(include=["number", "bool"]).copy()
    numeric = numeric.drop(columns=[TARGET_COLUMN, "row_count"], errors="ignore")
    correlations = numeric.corrwith(data[TARGET_COLUMN]).abs().dropna()
    selected = correlations.sort_values(ascending=False).head(limit).index.tolist()
    if selected:
        return selected

    return numeric.columns[:limit].tolist()


def plot_sensor_correlation_heatmap(data, selected_features):
    """Save a correlation heatmap for selected engineered sensor features."""
    if not selected_features:
        raise ValueError("No numeric engineered sensor features are available for correlation.")

    correlation = data[selected_features].corr()
    plt.figure(figsize=(11, 9))
    axis = sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        annot=False,
        square=True,
        linewidths=0.25,
        cbar_kws={"label": "Pearson correlation"},
    )
    axis.set_title("Correlation Heatmap of Target-Correlated Engineered Sensor Features")
    axis.set_xlabel("Engineered feature")
    axis.set_ylabel("Engineered feature")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(SENSOR_CORRELATION_HEATMAP, dpi=150)
    plt.close()


def event_split_lookup(split_info):
    """Map failure event IDs to the split that owns each event."""
    lookup = {}
    for split_name, key in [
        ("train", "train_events"),
        ("validation", "validation_events"),
        ("test", "test_events"),
    ]:
        for event_id in split_info[key]:
            lookup[event_id] = split_name
    return lookup


def plot_failure_windows_timeline(data, split_info):
    """Save an hourly timeline of normal and failure-risk windows."""
    hourly_counts = (
        data.set_index(WINDOW_START_COLUMN)
        .groupby([pd.Grouper(freq="1h"), TARGET_COLUMN])
        .size()
        .unstack(fill_value=0)
        .rename(columns=CLASS_LABELS)
    )
    for label in CLASS_LABELS.values():
        if label not in hourly_counts.columns:
            hourly_counts[label] = 0
    hourly_counts = hourly_counts[["normal", "failure-risk"]]

    fig, axis = plt.subplots(figsize=(13, 5))
    split_colors = {
        "train": "#DDEAF7",
        "validation": "#E8F3E4",
        "test": "#FCE5E5",
    }
    lookup = event_split_lookup(split_info)
    split_label_seen = set()
    for block in split_info["blocks"]:
        split_name = lookup[block["event_id"]]
        label = f"{split_name} block" if split_name not in split_label_seen else None
        axis.axvspan(
            block["block_start"],
            block["block_end"],
            color=split_colors[split_name],
            alpha=0.45,
            label=label,
        )
        split_label_seen.add(split_name)

    axis.plot(
        hourly_counts.index,
        hourly_counts["normal"],
        color="#4C78A8",
        linewidth=1.2,
        label="normal windows",
    )
    axis.plot(
        hourly_counts.index,
        hourly_counts["failure-risk"],
        color="#E45756",
        linewidth=1.6,
        label="failure-risk windows",
    )
    for start, end in FAILURE_RISK_WINDOWS:
        axis.axvline(pd.to_datetime(start), color="#E45756", linewidth=0.6, alpha=0.6)
        axis.axvline(pd.to_datetime(end), color="#E45756", linewidth=0.6, alpha=0.6)

    axis.set_title("Normal and Failure-Risk Windows Over Time")
    axis.set_xlabel("Window start time")
    axis.set_ylabel("1-minute window count per hour")
    axis.xaxis.set_major_locator(mdates.MonthLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(FAILURE_WINDOWS_TIMELINE_PLOT, dpi=150)
    plt.close()


def select_distribution_features(data, selected_features):
    """Pick a small set of readable key sensor features for class-wise distributions."""
    mean_features = [
        column
        for column in selected_features
        if column.endswith("_mean") and column in data.columns
    ]
    if len(mean_features) >= MAX_DISTRIBUTION_FEATURES:
        return mean_features[:MAX_DISTRIBUTION_FEATURES]

    filled = mean_features.copy()
    for column in selected_features:
        if column in filled or column == "row_count":
            continue
        filled.append(column)
        if len(filled) == MAX_DISTRIBUTION_FEATURES:
            break
    return filled


def plot_key_sensor_distributions(data, distribution_features):
    """Save class-wise boxplots for selected key sensor features."""
    if not distribution_features:
        raise ValueError("No numeric features are available for sensor distribution plots.")

    plot_data = data[[TARGET_COLUMN, *distribution_features]].copy()
    plot_data["target_class"] = plot_data[TARGET_COLUMN].map(CLASS_LABELS)

    rows = 2
    columns = 3
    fig, axes = plt.subplots(rows, columns, figsize=(14, 8))
    axes = axes.flatten()

    for index, feature in enumerate(distribution_features):
        axis = axes[index]
        sns.boxplot(
            data=plot_data,
            x="target_class",
            y=feature,
            hue="target_class",
            palette=["#4C78A8", "#E45756"],
            showfliers=False,
            legend=False,
            ax=axis,
        )
        axis.set_title(feature)
        axis.set_xlabel("Target class")
        axis.set_ylabel("Feature value")

    for axis in axes[len(distribution_features) :]:
        axis.axis("off")

    fig.suptitle("Key Sensor Feature Distributions by Target Class", y=0.98)
    plt.tight_layout()
    plt.savefig(KEY_SENSOR_DISTRIBUTIONS_PLOT, dpi=150)
    plt.close()


def build_eda_summary(data):
    """Create a report-ready summary table for the processed dataset."""
    X, _ = split_features_labels(data)
    start_time = data[WINDOW_START_COLUMN].min()
    end_time = data[WINDOW_START_COLUMN].max()
    class_balance = class_balance_frame(data)
    missing_counts = data.isna().sum()
    missing_nonzero = missing_counts[missing_counts > 0].sort_values(ascending=False)

    rows = [
        {"section": "dataset", "metric": "row_count", "value": len(data)},
        {"section": "dataset", "metric": "column_count", "value": data.shape[1]},
        {
            "section": "dataset",
            "metric": "dataset_shape",
            "value": f"{data.shape[0]} rows x {data.shape[1]} columns",
        },
        {
            "section": "features",
            "metric": "feature_count_excluding_target_and_window_start",
            "value": X.shape[1],
        },
        {"section": "date_range", "metric": "start", "value": start_time},
        {"section": "date_range", "metric": "end", "value": end_time},
    ]

    for _, row in class_balance.iterrows():
        rows.append(
            {
                "section": "class_counts",
                "metric": f"{row['class_label']}_count",
                "value": row["count"],
            }
        )
        rows.append(
            {
                "section": "class_percentages",
                "metric": f"{row['class_label']}_percentage",
                "value": round(row["percentage"], 6),
            }
        )

    rows.extend(
        [
            {
                "section": "missing_values",
                "metric": "total_missing_cells",
                "value": int(missing_counts.sum()),
            },
            {
                "section": "missing_values",
                "metric": "columns_with_missing_values",
                "value": int((missing_counts > 0).sum()),
            },
        ]
    )

    for column, count in missing_nonzero.items():
        rows.append(
            {
                "section": "missing_values_by_column",
                "metric": column,
                "value": int(count),
            }
        )

    return pd.DataFrame(rows)


def build_event_summary(data, split_info):
    """Create one event-level summary row for each configured failure event."""
    events = parse_failure_events()
    lookup = event_split_lookup(split_info)
    rows = []
    for event in events:
        risk_start = event["start"]
        failure_start = risk_start + FAILURE_LEAD_TIME
        failure_end = event["end"]
        event_mask = (
            (data[WINDOW_START_COLUMN] >= risk_start)
            & (data[WINDOW_START_COLUMN] <= failure_end)
            & (data[TARGET_COLUMN] == 1)
        )
        rows.append(
            {
                "event_id": event["event_id"],
                "split": lookup.get(event["event_id"], ""),
                "risk_start": risk_start,
                "failure_start": failure_start,
                "failure_end": failure_end,
                "positive_window_count": int(event_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def run_eda():
    """Generate EDA plots and CSV summaries from the processed windowed dataset."""
    ensure_directories([RESULTS_DIR, PLOTS_DIR])

    data = load_windowed_dataset(WINDOWED_LABELED_DATA_FILE)
    train_frame, validation_frame, test_frame, split_info = event_aware_blocked_split(data)

    selected_features = select_target_correlated_features(data)
    distribution_features = select_distribution_features(data, selected_features)

    plot_class_balance(data)
    plot_split_class_balance(train_frame, validation_frame, test_frame)
    plot_sensor_correlation_heatmap(data, selected_features)
    plot_failure_windows_timeline(data, split_info)
    plot_key_sensor_distributions(data, distribution_features)

    eda_summary = build_eda_summary(data)
    event_summary = build_event_summary(data, split_info)
    eda_summary.to_csv(EDA_SUMMARY_FILE, index=False)
    event_summary.to_csv(EVENT_SUMMARY_FILE, index=False)

    print(f"Processed dataset: {WINDOWED_LABELED_DATA_FILE}")
    print(f"Processed dataset shape: {data.shape}")
    print(f"Selected correlation features: {', '.join(selected_features)}")
    print(f"Selected distribution features: {', '.join(distribution_features)}")
    print(f"Saved plot to {CLASS_BALANCE_PLOT}")
    print(f"Saved plot to {SPLIT_CLASS_BALANCE_PLOT}")
    print(f"Saved plot to {SENSOR_CORRELATION_HEATMAP}")
    print(f"Saved plot to {FAILURE_WINDOWS_TIMELINE_PLOT}")
    print(f"Saved plot to {KEY_SENSOR_DISTRIBUTIONS_PLOT}")
    print(f"Saved EDA summary to {EDA_SUMMARY_FILE}")
    print(f"Saved event summary to {EVENT_SUMMARY_FILE}")
    return eda_summary, event_summary


def main():
    """Run EDA as a command-line module."""
    sns.set_theme(style="whitegrid")
    try:
        run_eda()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
