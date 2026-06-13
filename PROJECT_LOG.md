# Project Log: Failure-Risk Prediction for a Metro Train Compressor Using Machine Learning

This file records project decisions, commands, changes, problems, fixes, and reproducibility notes for the COEN 330 final report. Do not overwrite old entries. Append new entries whenever the project changes.

## 1. Project Overview

Project title: Failure-Risk Prediction for a Metro Train Compressor Using Machine Learning.

The project builds a machine learning workflow for predicting whether a 1-minute metro train compressor sensor window belongs to a failure-risk/anomaly state. Compressor failures in railway systems can cause high repair cost, service disruption, and operational risk, so the model should prioritize catching failure-risk windows over maximizing overall accuracy.

## 2. Dataset Choice and Reason

Dataset: MetroPT-3 Dataset from the UCI Machine Learning Repository.

The MetroPT-3 dataset was selected because it contains real air compressor sensor readings over time and is appropriate for predictive maintenance and anomaly/failure-risk classification. The raw CSV is stored locally in `data/raw/` and is not committed to GitHub because it is large.

Raw data file expected locally:

```text
data/raw/MetroPT3(AirCompressor).csv
```

Confirmed raw data after successful preprocessing:

- Raw shape: 1,516,948 rows and 17 columns
- Timestamp column: `timestamp`

## 3. Machine Learning Task

Task type: supervised binary classification.

The model should classify each 1-minute compressor sensor window as either normal operation or failure-risk/anomaly condition.

## 4. Target Label Definition

Target column:

- `0` = normal operation
- `1` = failure-risk/anomaly condition

Labeling rule: failure-risk is defined as one hour before each known failure start time until the failure end time. All other timestamps/windows are labeled normal. This creates a risk-state/early-warning label, not merely a current-failure label.

Status: Configured in `src/config.py` using the known failure windows. Each labeled failure-risk window starts one hour before the known failure start and ends at the known failure end.

## 5. Feature Engineering Plan

Raw time-series sensor readings are aggregated into 1-minute windows.

For each numeric sensor, compute:

- Mean
- Standard deviation
- Minimum
- Maximum
- Last value

Current implementation status: 1-minute windows with mean, standard deviation, minimum, maximum, last value, and row count have been implemented and confirmed by running `python -m src.preprocessing`.

## 6. Validation Strategy

Use an event-aware blocked time split:

- Training: early normal data plus the first two failure-risk events
- Validation: the third failure-risk event plus nearby normal data
- Test: the fourth/final failure-risk event plus nearby normal data

Reason for event-aware blocked splitting: MetroPT-3 is time-series data, so random row-level splitting could leak nearby correlated windows between training, validation, and test sets. A strict chronological 70/15/15 split was tested but rejected for the final methodology because the final 15% contained 0 failure-risk windows, making positive-class recall, F2-score, F1-score, and the confusion matrix unsuitable for a failure-risk prediction project.

The event-aware blocked split uses `window_start` and configured failure-risk windows to assign whole failure events to splits without leaking the same failure event across train, validation, and test. `window_start` is not used as a model feature.

Status: Implemented in `src/modeling.py`, `src/train.py`, and `src/evaluate.py`.

## 7. Models Selected

Planned models:

1. Logistic Regression - baseline
2. Decision Tree
3. Random Forest
4. Extra Trees
5. HistGradientBoostingClassifier

Status: Implemented in `src/train.py` for the final model comparison.

## 8. Metrics Selected

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- F2-score
- ROC-AUC when possible
- Confusion matrix

Primary metric emphasis:

- Recall
- F2-score
- F1-score
- Confusion matrix

Accuracy is reported but should not be used as the main success metric.

Reason: the confirmed processed dataset is highly imbalanced. Accuracy alone can be misleading because the normal class is much larger than the failure-risk class. False negatives are especially costly in predictive maintenance because a missed failure-risk window can mean a missed opportunity to inspect, repair, or plan service before compressor failure. Recall measures how many true failure-risk windows are caught, F2-score weights recall more heavily than precision, and F1-score summarizes the precision/recall tradeoff.

Status: Verified model results generated using the event-aware blocked split.

## 9. Repository and GitHub Setup

Remote repository:

```text
https://github.com/arnav-singh-ahlawat/coen330_project.git
```

Important Git rule: Do not commit raw data files, ZIP files, virtual environments, or large model files. Keep raw data local in `data/raw/`. Commit only code, README, requirements, report notes, demo, metrics, and plots.

## 10. Commands Run

Commands already run:

```bash
git init
git branch -m main
git remote add origin https://github.com/arnav-singh-ahlawat/coen330_project.git
git push -u origin main
python -m src.preprocessing
```

Recent verification commands:

```bash
python -m compileall src
python -m src.preprocessing
```

Result: `python -m compileall src` passed. After the failure windows were configured, `python -m src.preprocessing` created `data/processed/windowed_labeled_data.csv`.

Confirmed preprocessing run:

```bash
python -m src.preprocessing
```

Confirmed input and output:

- Raw input file: `data/raw/MetroPT3(AirCompressor).csv`
- Raw shape: 1,516,948 rows and 17 columns
- Timestamp column: `timestamp`
- Processed output file: `data/processed/windowed_labeled_data.csv`
- Final processed shape: 252,720 rows and 78 columns
- Processed file size: about 168 MB

## 10.1 Preprocessing and Labeling Verification

Verification run before threshold tuning:

```bash
python -m compileall src
python - <<'PY'
# audited processed labels, configured failure windows, model features, and event split
PY
```

Dataset documentation check: the UCI MetroPT-3 page lists four failure intervals:

- April 18, 2020 00:00 to April 18, 2020 23:59
- May 29, 2020 23:30 to May 30, 2020 06:00
- June 5, 2020 10:00 to June 7, 2020 14:30
- July 15, 2020 14:30 to July 15, 2020 19:00

The project labeling implementation imports `FAILURE_RISK_WINDOWS` from `src/config.py` in `src/labeling.py`. The configured risk windows match the documented failure intervals with the planned one-hour lead time:

| Event | risk_start | failure_start | failure_end | Positive 1-minute windows | First positive window | Last positive window | Verified |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| 1 | 2020-04-17 23:00:00 | 2020-04-18 00:00:00 | 2020-04-18 23:59:00 | 1,496 | 2020-04-17 23:00:00 | 2020-04-18 23:59:00 | Yes |
| 2 | 2020-05-29 22:30:00 | 2020-05-29 23:30:00 | 2020-05-30 06:00:00 | 451 | 2020-05-29 22:30:00 | 2020-05-30 06:00:00 | Yes |
| 3 | 2020-06-05 09:00:00 | 2020-06-05 10:00:00 | 2020-06-07 14:30:00 | 2,923 | 2020-06-05 09:00:00 | 2020-06-07 14:19:00 | Yes |
| 4 | 2020-07-15 13:30:00 | 2020-07-15 14:30:00 | 2020-07-15 19:00:00 | 330 | 2020-07-15 13:30:00 | 2020-07-15 19:00:00 | Yes |

Label application check:

- `target` was recomputed from the processed `window_start` column using `label_from_failure_windows`.
- Mismatches between recomputed labels and saved labels: 0.
- Total positive windows in `data/processed/windowed_labeled_data.csv`: 5,200.
- This confirms target labels are applied to 1-minute windows using `window_start`.

Feature exclusion check:

- `window_start` in model features: False.
- `target` in model features: False.
- Model feature count after exclusions: 76.

Event-aware blocked split verification:

| Split | Rows | Date range | Class 0 | Class 1 | Failure event IDs |
| --- | ---: | --- | ---: | ---: | --- |
| Train | 144,412 | 2020-02-01 00:00:00 to 2020-06-02 07:29:00 | 142,465 | 1,947 | 1, 2 |
| Validation | 30,721 | 2020-06-02 07:30:00 to 2020-06-26 13:59:00 | 27,798 | 2,923 | 3 |
| Test | 77,587 | 2020-06-26 14:00:00 to 2020-09-01 03:59:00 | 77,257 | 330 | 4 |

Confirmed split assignment:

- Train uses failure events 1 and 2.
- Validation uses failure event 3.
- Test uses failure event 4.
- No failure event is shared across train, validation, and test.

## 11. Problems Encountered

- `python src/preprocessing.py` originally ran with no useful output and did not clearly show whether `data/processed/windowed_labeled_data.csv` was created.
- `src/preprocessing.py` originally only defined an sklearn numeric preprocessing pipeline and did not run the data preparation workflow.
- `FAILURE_RISK_WINDOWS` was empty in `src/config.py`, so valid supervised labels could not be generated from failure windows and `data/processed/windowed_labeled_data.csv` remained absent.
- The timestamp column must be detected reliably before time-window aggregation.
- Earlier final model results did not exist and could not be invented; verified results now exist after the corrected training/evaluation run.
- Running training as `python src/train.py` failed with `ModuleNotFoundError: No module named 'src'`.
- Suspected training/evaluation issue: `src/train.py` and `src/evaluate.py` were rebuilding labels from the raw CSV and using a random `train_test_split` instead of loading `data/processed/windowed_labeled_data.csv`.
- The reported classification support count of 303,390 was suspicious because the confirmed processed windowed dataset has 252,720 rows. A support count of 303,390 is about 20% of the original 1,516,948 raw records, matching the previous random test split size and indicating the raw dataset was likely being evaluated.
- Strict chronological 70/15/15 splitting was also unsuitable for the final evaluation because the final 15% test split contained 37,908 normal windows and 0 failure-risk windows.
- After preprocessing, labeling, and event-aware splitting were verified, the remaining modeling issue was generalization to held-out event 4. The previously selected model did not identify any of the 330 failure-risk windows in event 4.

## 12. Fixes Applied

- Added a runnable preprocessing workflow in `src/preprocessing.py`.
- Added timestamp detection in `src/data_loading.py`.
- Added 1-minute window feature creation in `src/features.py`.
- Added last-value aggregation to match the planned 1-minute feature engineering method.
- Added `WINDOWED_LABELED_DATA_FILE` in `src/config.py`.
- Added explicit progress messages and final dataset reporting to preprocessing.
- Added a clear error path when failure windows are missing.
- Added a clear error path when a timestamp column cannot be found.
- Configured the known failure-risk windows in `src/config.py`, with each risk interval beginning one hour before failure start.
- Updated `src/preprocessing.py` so terminal execution calls `main()`, parses the detected timestamp column as datetime, aggregates 1-minute features, labels the windowed data, and writes `data/processed/windowed_labeled_data.csv`.
- Documented module-mode execution from the repository root: `python -m src.preprocessing`, `python -m src.train`, and `python -m src.evaluate`.
- Confirmed `src/__init__.py` exists so `src` is a Python package.
- Cause of the training error: running `src/train.py` as a script puts `src/` itself on `sys.path`, so absolute imports like `from src...` cannot resolve the package. Running with `python -m src.train` from the repository root keeps the project root on `sys.path`.
- Fix: use `python -m src.train` instead of `python src/train.py`.
- Reworked `src/train.py` so it loads only `data/processed/windowed_labeled_data.csv` for the final project pipeline.
- Added `src/modeling.py` with shared loading, event-aware blocked splitting, feature/label separation, class balance reporting, and metric calculation.
- Reworked `src/evaluate.py` so it loads the saved final model and evaluates it against the event-aware blocked test split from the processed windowed dataset.
- Updated `src/config.py` so the final model path is `models/final_model.joblib` and final test metrics are saved to `results/test_metrics.csv`.
- Ensured `window_start` is used only to sort and split records by time, then dropped before model fitting/evaluation.
- Replaced random row-level splitting and strict 70/15/15 chronological splitting with the current event-aware blocked split.
- Assigned failure events by configured failure-risk windows: events 1 and 2 are used for training, event 3 is used for validation, and event 4 is used for testing.
- Added normal windows around each event block so train, validation, and test each contain both normal and failure-risk samples.
- Implemented the required five-model comparison: Logistic Regression baseline, Decision Tree, Random Forest, Extra Trees, and HistGradientBoostingClassifier.
- Added `class_weight="balanced"` for Logistic Regression, Decision Tree, Random Forest, and Extra Trees.
- Added small-grid tuning for Decision Tree, Random Forest, and HistGradientBoostingClassifier.
- Added validation-only threshold tuning for all five fitted models using class-1 probabilities and thresholds from 0.01 through 0.99.
- For each model, selected one threshold that maximizes validation F1-score and one threshold that maximizes validation F2-score.
- Compared three model-selection strategies: best validation F1 at default threshold 0.50, best validation F1 after threshold tuning, and best validation F2 after threshold tuning.
- Final model selection uses the validation-tuned F1 strategy. Test data is not used for model or threshold selection.
- The held-out test split is evaluated once after model and threshold selection.

## 13. Results Generated

Model metrics: Generated from `python -m src.train`.

### Demo Repair

Demo error:

```bash
python demo/demo.py
python -m demo.demo
```

Both demo entrypoints failed with:

```text
ValueError: The feature names should match those that were passed during fit.
```

Cause: `demo/demo.py` rebuilt raw-style feature columns such as `caudal_impulses`,
`comp`, `dayofweek`, `dv_eletric`, and `dv_pressure`. The final model was trained on
the processed 1-minute window feature table, where the model features exclude
`window_start` and `target` and include engineered columns such as
`caudal_impulses_mean`, `caudal_impulses_std`, `caudal_impulses_min`,
`caudal_impulses_max`, and `caudal_impulses_last`.

Fix: updated `demo/demo.py` so it loads `data/processed/windowed_labeled_data.csv`,
loads `models/final_model.joblib`, derives features with the same training helper that
drops `window_start` and `target`, selects a normal sample and a failure-risk sample
when available, uses `predict_proba`, and applies the saved final threshold from
`results/threshold_table.csv`. If the threshold table is missing, the demo warns and
defaults to threshold `0.50`.

Verification commands passed:

```bash
python demo/demo.py
python -m demo.demo
```

Plots: Generated from `python -m src.train` and `python -m src.evaluate`.

Processed windowed labeled dataset: Confirmed. Local rerun created `data/processed/windowed_labeled_data.csv` with 252,720 rows and 78 columns.

Confirmed class balance:

- Class 0 normal: 247,520 windows, 97.94%
- Class 1 failure-risk/anomaly: 5,200 windows, 2.06%

Interpretation: the dataset is highly imbalanced. Accuracy alone is not a reliable main metric. The evaluation should emphasize recall, F2-score, F1-score, and the confusion matrix for the failure-risk class, while also reporting precision, ROC-AUC when possible, and accuracy as secondary context.

Confirmed current behavior before the latest fix: preprocessing did not create fake labels or fake results. It exited with a clear missing-failure-windows error.

Current run commands:

```bash
python -m src.preprocessing
python -m src.train
python -m src.evaluate
python -m src.stratified_baseline
python -m src.eda
```

Rejected strict chronological 70/15/15 split:

- Train: 176,904 windows
- Validation: 37,908 windows
- Test: 37,908 windows
- Final test class balance: 37,908 normal windows and 0 failure-risk windows

Reason rejected: the final 15% contained no positive class support, so positive-class recall, F2-score, F1-score, and the confusion matrix could not meaningfully evaluate failure-risk prediction.

Current event-aware blocked split from the processed dataset:

- Train: 144,412 windows; class 0 = 142,465, class 1 = 1,947
- Validation: 30,721 windows; class 0 = 27,798, class 1 = 2,923
- Test: 77,587 windows; class 0 = 77,257, class 1 = 330
- Train failure events: events 1 and 2
- Validation failure event: event 3
- Test failure event: event 4

This split avoids random row-level leakage and keeps the held-out test failure event out of training and validation.

Verified validation threshold comparison from `results/threshold_table.csv`:

| Model | Default F1 at 0.50 | Best F1 threshold | Best validation F1 | Best F2 threshold | Best validation F2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression baseline | 0.7550 | 0.61 | 0.8216 | 0.61 | 0.8945 |
| Decision Tree | 0.5001 | 0.13 | 0.5001 | 0.13 | 0.3856 |
| Random Forest | 0.5313 | 0.31 | 0.5525 | 0.19 | 0.4790 |
| Extra Trees | 0.0409 | 0.04 | 0.6952 | 0.03 | 0.7717 |
| HistGradientBoostingClassifier | 0.6603 | 0.11 | 0.6652 | 0.11 | 0.5595 |

Model selection strategy comparison:

- Best default-threshold validation F1: Logistic Regression baseline, threshold 0.50, F1 = 0.7550.
- Best tuned-threshold validation F1: Logistic Regression baseline, threshold 0.61, F1 = 0.8216.
- Best tuned-threshold validation F2: Logistic Regression baseline, threshold 0.61, F2 = 0.8945.

Selected final model: Logistic Regression baseline, selected by validation-tuned F1-score using threshold 0.61.

Validation confusion matrix for the selected final model at threshold 0.61:

- True normal predicted normal: 26,735
- True normal predicted failure-risk: 1,063
- True failure-risk predicted normal: 144
- True failure-risk predicted failure-risk: 2,779

Validation predicted class counts for the selected final model at threshold 0.61:

- Predicted normal: 26,879
- Predicted failure-risk: 3,842

Validation class-1 probability summary for the selected final model:

- Minimum: 0.000000
- Mean: 0.147957
- Maximum: 1.000000

Verified final test metrics from `results/test_metrics.csv`:

- Threshold selected on validation: 0.61
- Accuracy: 0.9942902805882429
- Precision: 0.0
- Recall: 0.0
- F1-score: 0.0
- F2-score: 0.0
- ROC-AUC: 0.12087550368094527

Final test confusion matrix counts for labels `[normal, failure_risk]`:

- True normal predicted normal: 77,144
- True normal predicted failure-risk: 113
- True failure-risk predicted normal: 330
- True failure-risk predicted failure-risk: 0

Final test predicted class counts:

- Predicted normal: 77,474
- Predicted failure-risk: 113

Final test class-1 probability summary:

- Minimum: 0.000000
- Mean: 0.024688
- Maximum: 0.999985

Interpretation: threshold tuning improved validation F1 for the selected model, but it still did not recover failure-risk windows in held-out event 4. This indicates the remaining issue is not preprocessing or split leakage; it is poor generalization from training events 1-2 and validation event 3 to the final test event.

Report interpretation: the event-aware held-out test result should be presented as a major limitation of supervised learning with only four independent failure events. The high accuracy is caused by the large number of normal windows and does not mean the model is successful. For the failure-risk class, the final test result has recall = 0.0, F2-score = 0.0, F1-score = 0.0, 330 false negatives, and 0 true positives. This is not a coding error to hide; it is the central evaluation finding and means the current model is not deployment-ready.

Secondary stratified window-level baseline:

- Script: `src/stratified_baseline.py`
- Command: `python -m src.stratified_baseline`
- Input: `data/processed/windowed_labeled_data.csv`
- Split: stratified random train/test split over processed windows, test size = 20%, random state = 42
- Train: 202,176 windows; class 0 = 198,016, class 1 = 4,160
- Test: 50,544 windows; class 0 = 49,504, class 1 = 1,040
- Features: 76 numeric engineered features after dropping `target` and `window_start`

Stratified baseline metrics from `results/stratified_baseline_metrics.csv`:

| Model | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | Confusion matrix `[tn, fp, fn, tp]` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression baseline | 0.9636 | 0.3573 | 0.9615 | 0.5210 | 0.7185 | 0.9849 | `[47705, 1799, 40, 1000]` |
| Decision Tree | 0.9951 | 0.8421 | 0.9385 | 0.8877 | 0.9175 | 0.9687 | `[49321, 183, 64, 976]` |
| Random Forest | 0.9963 | 0.8638 | 0.9760 | 0.9165 | 0.9513 | 0.9950 | `[49344, 160, 25, 1015]` |
| Extra Trees | 0.9963 | 0.8695 | 0.9673 | 0.9158 | 0.9460 | 0.9926 | `[49353, 151, 34, 1006]` |
| HistGradientBoostingClassifier | 0.9955 | 0.8301 | 0.9817 | 0.8996 | 0.9471 | 0.9986 | `[49295, 209, 19, 1021]` |

Interpretation of the stratified baseline: this is an optimistic comparison only. The dataset has only four independent failure events, and the 5,200 positive windows are not 5,200 independent compressor failures. A stratified window-level split mixes windows from the same failure events across train and test, so it can show that engineered features separate labeled risk windows when event independence is not enforced, but it can also overestimate predictive-maintenance performance. The event-aware split remains the realistic deployment-oriented result because it tests whether training on failure events 1 and 2 and selecting on event 3 generalizes to held-out failure event 4.

Saved outputs:

- `results/metrics_table.csv`
- `results/threshold_table.csv`
- `models/final_model.joblib`
- `results/test_metrics.csv`
- `results/training_output.txt`
- `results/plots/confusion_matrix.png`
- `results/plots/validation_probability_distribution.png`
- `results/plots/test_probability_distribution.png`
- `results/plots/precision_recall_curve.png`
- `results/stratified_baseline_metrics.csv`
- `results/plots/stratified_baseline_confusion_matrix.png`

EDA outputs generated from `python -m src.eda`:

- `results/eda_summary.csv`
- `results/event_summary.csv`
- `results/plots/class_balance.png`
- `results/plots/split_class_balance.png`
- `results/plots/sensor_correlation_heatmap.png`
- `results/plots/failure_windows_timeline.png`
- `results/plots/key_sensor_distributions.png`

Confirmed EDA summary:

- Dataset shape: 252,720 rows and 78 columns
- Model feature count after dropping `target` and `window_start`: 76
- Date range: 2020-02-01 00:00:00 to 2020-09-01 03:59:00
- Class 0 normal: 247,520 windows, 97.942387%
- Class 1 failure-risk: 5,200 windows, 2.057613%
- Missing values: 1,470 total missing cells across 15 columns; the missing columns are the sensor standard-deviation features with 98 missing values each.

Confirmed event summary:

| Event | Split | Risk start | Failure start | Failure end | Positive windows |
| --- | --- | --- | --- | --- | ---: |
| 1 | Train | 2020-04-17 23:00:00 | 2020-04-18 00:00:00 | 2020-04-18 23:59:00 | 1,496 |
| 2 | Train | 2020-05-29 22:30:00 | 2020-05-29 23:30:00 | 2020-05-30 06:00:00 | 451 |
| 3 | Validation | 2020-06-05 09:00:00 | 2020-06-05 10:00:00 | 2020-06-07 14:30:00 | 2,923 |
| 4 | Test | 2020-07-15 13:30:00 | 2020-07-15 14:30:00 | 2020-07-15 19:00:00 | 330 |

EDA interpretation for the report:

- The processed dataset is highly imbalanced: 97.94% normal windows and 2.06% failure-risk windows.
- The positive class contains 5,200 labeled failure-risk windows, but those windows come from only 4 independent failure events.
- The class-balance and split-balance plots confirm why accuracy is misleading: a model can classify most windows as normal and still appear accurate while missing the rare class.
- The EDA supports reporting recall, F2-score, F1-score, and the confusion matrix as primary evidence because these metrics show missed failure-risk windows directly.
- The timeline plot shows that positive windows occur in four separated event clusters, supporting the event-aware split decision instead of a random window-level split.

## 14. Report Notes

The final report should explain:

- Why MetroPT-3 is appropriate for predictive maintenance.
- Why the task is supervised binary classification.
- How labels are defined from known failure windows.
- Why 1-minute windows are used.
- Why failure-risk prediction matters in railway maintenance.
- Why false negatives are more costly than false positives.
- Why recall, F2-score, F1-score, and the confusion matrix are emphasized.
- Why event-aware validation is more realistic than random row-level splitting.
- Why stratified window-level evaluation can be misleading for predictive maintenance.
- Why the confirmed 97.94% normal / 2.06% failure-risk class balance makes accuracy insufficient by itself.
- Why only four independent failure events is a major dataset limitation.
- Why the final event-aware test result demonstrates poor generalization to unseen failure event 4.
- Why the stratified baseline is included only as an optimistic feature-separability comparison, not as deployment performance.

Detailed report-ready notes are collected in `report/REPORT_NOTES.md`.

## 15. Next Steps

- Keep the report framing focused on failure-risk prediction for railway compressor predictive maintenance.
- Discuss the final model's poor held-out event recall: the selected model had strong validation F1 but did not identify any positive windows in the final failure event.
- Consider future blocked cross-validation across failure events if more labeled events become available.

## Future Update Table

| Date | Person | Action | Reason | Files Changed | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-12 | Niraj | Created project tracking and report notes files; updated README and final report draft references. | Keep project decisions and reproducibility notes organized for the COEN 330 report. | `PROJECT_LOG.md`, `report/REPORT_NOTES.md`, `README.md`, `report/final_report_draft.md` | Confirmed after file creation. | Results remain pending until metrics are generated. |
| 2026-06-12 | Niraj | Converted preprocessing into a runnable data preparation script with timestamp detection and 1-minute window features. | `python src/preprocessing.py` previously ran without clear output and did not confirm processed data creation. | `src/preprocessing.py`, `src/data_loading.py`, `src/features.py`, `src/config.py` | Confirmed error path: preprocessing exits clearly because failure windows are missing. | No fake labels or model results generated. |
| 2026-06-12 | Niraj | Added last-value aggregation to 1-minute window features. | Match the documented feature engineering plan for each numeric sensor. | `src/features.py`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md` | Confirmed. `python -m compileall src` passed. `python -m src.preprocessing` exits clearly because failure windows are missing. | Full processed output remains pending until failure windows are configured. |
| 2026-06-12 | Niraj | Configured failure-risk labeling windows and fixed terminal preprocessing execution. | `python -m src.preprocessing` ran but `data/processed` stayed empty because failure windows were not configured. | `src/config.py`, `src/preprocessing.py`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md` | Confirmed. `python -m src.preprocessing` created `data/processed/windowed_labeled_data.csv`. | Output was pending until rerun; no raw data, model files, or fake model results added. |
| 2026-06-12 | Niraj | Recorded confirmed preprocessing output and class balance. | Preprocessing now runs successfully and produced real processed dataset statistics for the report. | `PROJECT_LOG.md`, `report/REPORT_NOTES.md`, `report/final_report_draft.md` | Confirmed raw shape: 1,516,948 x 17. Confirmed processed shape: 252,720 x 78. Confirmed class balance: 247,520 normal windows and 5,200 failure-risk/anomaly windows. | Dataset is highly imbalanced; emphasize recall, F2-score, F1-score, and the confusion matrix. No raw or processed CSV files should be committed. |
| 2026-06-12 | Niraj | Fixed run instructions for package imports. | `python src/train.py` failed with `ModuleNotFoundError: No module named 'src'` because script execution from inside `src/` does not resolve absolute `src` imports. | `README.md`, `PROJECT_LOG.md`, `src/__init__.py` | Confirmed documented commands are `python -m src.preprocessing`, `python -m src.train`, and `python -m src.evaluate`. | No ML methodology changed. |
| 2026-06-12 | Niraj | Reworked training and evaluation around the processed windowed dataset and chronological split. | The previous training/evaluation path could rebuild from raw data and produced a suspicious 303,390 support count, about 20% of the raw dataset rather than matching the 252,720-row processed dataset. | `src/config.py`, `src/modeling.py`, `src/train.py`, `src/evaluate.py`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md` | `python -m compileall src`, `python -m src.train`, and `python -m src.evaluate` passed. Generated `results/metrics_table.csv`, `models/final_model.joblib`, `results/test_metrics.csv`, and `results/training_output.txt`. | Intermediate 70/15/15 chronological methodology was later superseded because its final test split had 0 failure-risk windows. |
| 2026-06-12 | Niraj | Replaced strict chronological 70/15/15 with event-aware blocked splitting. | The final 15% chronological test split contained 0 failure-risk windows, making positive-class recall, F2-score, F1-score, and confusion matrix unsuitable. | `src/config.py`, `src/modeling.py`, `src/train.py`, `src/evaluate.py`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md` | `python -m compileall src`, `python -m src.train`, and `python -m src.evaluate` passed. Event-aware test split now contains 330 failure-risk windows from held-out event 4. | Avoids random row-level leakage and avoids leaking the same failure event across train, validation, and test. |
| 2026-06-12 | Niraj | Verified preprocessing and labeling before threshold tuning. | Confirm labels, feature exclusions, and event-aware split correctness before changing decision thresholds. | `PROJECT_LOG.md`, `report/REPORT_NOTES.md` | Confirmed UCI failure times match configured failure windows with one-hour risk lead, recomputed labels from `window_start` with 0 mismatches, confirmed `window_start` and `target` are excluded from features, and confirmed split events are train 1-2, validation 3, test 4. | No model code or threshold tuning changed. |
| 2026-06-12 | Niraj | Added validation-only threshold tuning and probability diagnostics. | Improve modeling diagnostics fairly after preprocessing and labels were verified, without tuning on the held-out test event. | `src/config.py`, `src/modeling.py`, `src/train.py`, `src/evaluate.py`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md` | `python -m compileall src`, `python -m src.train`, and `python -m src.evaluate` passed. Generated threshold table, probability plots, validation precision-recall curve, and thresholded test metrics. | Final selection used validation-tuned F1 only. Test event 4 still had 0 true positives, so the remaining issue is held-out event generalization. |
| 2026-06-12 | Niraj | Reframed documentation around failure-risk prediction for metro train compressor predictive maintenance. | Align README, project log, report notes, final report draft, and script descriptions with the current problem statement and verified event-aware result. | `README.md`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md`, `report/final_report_draft.md`, `report/report_outline.md`, `src/config.py`, `src/train.py`, `src/evaluate.py`, `src/modeling.py`, `src/features.py`, `src/labeling.py`, `src/preprocessing.py`, `demo/README.md`, `demo/demo.py` | Documentation now emphasizes recall, F2-score, F1-score, confusion matrix, event-aware validation, and poor generalization to held-out event 4. | No labels, failure windows, test split, model selection results, or test metrics were changed. |
| 2026-06-12 | Niraj | Added a secondary stratified window-level baseline experiment. | Show how much performance can improve when engineered windows are split randomly without enforcing failure-event independence. | `src/stratified_baseline.py`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md`, `report/final_report_draft.md` | `python -m compileall src` and `python -m src.stratified_baseline` passed. Generated `results/stratified_baseline_metrics.csv` and `results/plots/stratified_baseline_confusion_matrix.png`. | Event-aware evaluation remains the realistic result; the stratified baseline is optimistic and not deployment performance. |
| 2026-06-13 | Niraj | Added EDA script, plots, and summary tables. | Provide report-ready exploratory evidence for class imbalance, event-level positives, metric choice, and event-aware splitting. | `src/eda.py`, `results/eda_summary.csv`, `results/event_summary.csv`, `results/plots/class_balance.png`, `results/plots/split_class_balance.png`, `results/plots/sensor_correlation_heatmap.png`, `results/plots/failure_windows_timeline.png`, `results/plots/key_sensor_distributions.png`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md`, `report/final_report_draft.md` | `python -m compileall src` and `python -m src.eda` passed. Confirmed 252,720 rows, 76 model features, 5,200 positive windows, and 4 independent failure events. | Training and evaluation were not changed. Raw and processed CSV files remain ignored and should not be committed. |
| 2026-06-13 | Niraj | Fixed the command-line demo feature mismatch. | The demo passed raw-style sensor columns to a model trained on processed 1-minute engineered window features. | `demo/demo.py`, `demo/README.md`, `README.md`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md` | Passed: `python demo/demo.py` and `python -m demo.demo`. | Training, evaluation, labels, failure windows, split, and generated CSV files were not changed. |
| 2026-06-13 | Niraj | Added PDF-based guideline checklist, repository audit, and updated future-Codex instructions. | The official COEN 330 guideline PDF was added and should be treated as the project requirements source of truth. | `docs/guideline_checklist.md`, `docs/guideline_audit.md`, `AGENTS.md`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md` | Extracted the PDF requirements, created a practical checklist, audited the repository against the PDF, and preserved the existing event-aware methodology rules. | Documentation-only change. Training, evaluation, labels, failure windows, split logic, raw/processed CSV files, and model artifacts were not changed. |
| 2026-06-13 | Niraj | Filled remaining report/submission documentation gaps in the draft. | Address guideline audit gaps without changing training, labels, failure windows, split logic, model selection, or reported metrics. | `report/final_report_draft.md`, `report/REPORT_NOTES.md`, `PROJECT_LOG.md`, `docs/guideline_audit.md` | Reorganized the draft into the 15 required sections; added five team-member contribution placeholders, references, source/tool acknowledgments, feature-interpretation limitations, outlier/missing-value discussion, and qualitative error analysis for event 4 false negatives and false positives. | Documentation-only change. Event-aware evaluation remains the main realistic result; stratified baseline remains optimistic only; event 4 generalization failure and non-deployment-ready status remain explicit. |
