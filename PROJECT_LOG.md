# Project Log: Metro Train Compressor Failure Detection Using Machine Learning

This file records project decisions, commands, changes, problems, fixes, and reproducibility notes for the COEN 330 final report. Do not overwrite old entries. Append new entries whenever the project changes.

## 1. Project Overview

Project title: Metro Train Compressor Failure Detection Using Machine Learning.

The project builds a machine learning workflow for detecting failure-risk/anomaly conditions in a metro train air compressor using sensor time-series data. The workflow is intended to support a supervised binary classification task and produce reproducible code, report notes, metrics, and plots for COEN 330.

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

The model should classify each timestamp/window as either normal operation or failure-risk/anomaly condition.

## 4. Target Label Definition

Target column:

- `0` = normal operation
- `1` = failure-risk/anomaly condition

Labeling rule: failure-risk is defined as one hour before each known failure start time until the failure end time. All other timestamps/windows are labeled normal.

Status: Configured in `src/config.py` using the known failure windows. Each labeled failure-risk window starts one hour before the known failure start and ends at the known failure end.

## 5. Feature Engineering Plan

Raw time-series sensor readings are aggregated into 1-minute windows.

For each numeric sensor, compute:

- Mean
- Standard deviation
- Minimum
- Maximum
- Last value

Current implementation status: 1-minute windows with mean, standard deviation, minimum, maximum, last value, and row count have been implemented and confirmed by running `python src/preprocessing.py`.

## 6. Validation Strategy

Use a time-based split:

- Earliest 70% = training
- Next 15% = validation
- Final 15% = test

Reason for time-based split: MetroPT-3 is time-series data, so random splitting could leak nearby time patterns between training and test sets. A chronological split better evaluates whether the model generalizes to future time periods.

Status: Pending implementation.

## 7. Models Selected

Planned models:

1. Logistic Regression - baseline
2. Decision Tree
3. Random Forest
4. Extra Trees
5. HistGradientBoostingClassifier

Status: Logistic Regression baseline exists in the training code. Additional models are pending.

## 8. Metrics Selected

Metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC when possible
- Confusion matrix

Main metric: F1-score.

Secondary metric: recall.

Reason: the confirmed processed dataset is highly imbalanced. Accuracy alone can be misleading because the normal class is much larger than the failure-risk class. F1-score and recall should be emphasized, especially recall for the failure-risk class.

Status: Pending verified model results.

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
python src/preprocessing.py
```

Recent verification commands:

```bash
python -m compileall src
python src/preprocessing.py
```

Result: `python -m compileall src` passed. After the failure windows were configured, `python src/preprocessing.py` created `data/processed/windowed_labeled_data.csv`.

Confirmed preprocessing run:

```bash
python src/preprocessing.py
```

Confirmed input and output:

- Raw input file: `data/raw/MetroPT3(AirCompressor).csv`
- Raw shape: 1,516,948 rows and 17 columns
- Timestamp column: `timestamp`
- Processed output file: `data/processed/windowed_labeled_data.csv`
- Final processed shape: 252,720 rows and 78 columns
- Processed file size: about 168 MB

## 11. Problems Encountered

- `python src/preprocessing.py` originally ran with no useful output and did not clearly show whether `data/processed/windowed_labeled_data.csv` was created.
- `src/preprocessing.py` originally only defined an sklearn numeric preprocessing pipeline and did not run the data preparation workflow.
- `FAILURE_RISK_WINDOWS` was empty in `src/config.py`, so valid supervised labels could not be generated from failure windows and `data/processed/windowed_labeled_data.csv` remained absent.
- The timestamp column must be detected reliably before time-window aggregation.
- Final model results do not exist yet and must not be invented.

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

## 13. Results Generated

Model metrics: Pending.

Plots: Pending.

Processed windowed labeled dataset: Confirmed. Local rerun created `data/processed/windowed_labeled_data.csv` with 252,720 rows and 78 columns.

Confirmed class balance:

- Class 0 normal: 247,520 windows, 97.94%
- Class 1 failure-risk/anomaly: 5,200 windows, 2.06%

Interpretation: the dataset is highly imbalanced. Accuracy alone is not a reliable main metric. The evaluation should emphasize F1-score and recall, especially recall for the failure-risk class, while also reporting precision, ROC-AUC when possible, and the confusion matrix.

Confirmed current behavior before the latest fix: preprocessing did not create fake labels or fake results. It exited with a clear missing-failure-windows error.

## 14. Report Notes

The final report should explain:

- Why MetroPT-3 is appropriate for predictive maintenance.
- Why the task is supervised binary classification.
- How labels are defined from known failure windows.
- Why 1-minute windows are used.
- Why time-based splitting is used instead of random splitting.
- Why the confirmed 97.94% normal / 2.06% failure-risk class balance makes accuracy insufficient by itself.
- Why F1-score and recall matter for the failure-risk class.
- Which results are verified and which are pending.

Detailed report-ready notes are collected in `report/REPORT_NOTES.md`.

## 15. Next Steps

- Implement the 70/15/15 chronological train/validation/test split.
- Train and compare Logistic Regression, Decision Tree, Random Forest, Extra Trees, and HistGradientBoostingClassifier.
- Generate verified metrics and confusion matrix plots.
- Update `PROJECT_LOG.md`, `report/REPORT_NOTES.md`, and `report/final_report_draft.md` with confirmed results.

## Future Update Table

| Date | Person | Action | Reason | Files Changed | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-06-12 | Codex | Created project tracking and report notes files; updated README and final report draft references. | Keep project decisions and reproducibility notes organized for the COEN 330 report. | `PROJECT_LOG.md`, `report/REPORT_NOTES.md`, `README.md`, `report/final_report_draft.md` | Confirmed after file creation. | Results remain pending until metrics are generated. |
| 2026-06-12 | Codex | Converted preprocessing into a runnable data preparation script with timestamp detection and 1-minute window features. | `python src/preprocessing.py` previously ran without clear output and did not confirm processed data creation. | `src/preprocessing.py`, `src/data_loading.py`, `src/features.py`, `src/config.py` | Confirmed error path: preprocessing exits clearly because failure windows are missing. | No fake labels or model results generated. |
| 2026-06-12 | Codex | Added last-value aggregation to 1-minute window features. | Match the documented feature engineering plan for each numeric sensor. | `src/features.py`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md` | Confirmed. `python -m compileall src` passed. `python src/preprocessing.py` exits clearly because failure windows are missing. | Full processed output remains pending until failure windows are configured. |
| 2026-06-12 | Codex | Configured failure-risk labeling windows and fixed terminal preprocessing execution. | `python src/preprocessing.py` ran but `data/processed` stayed empty because failure windows were not configured. | `src/config.py`, `src/preprocessing.py`, `PROJECT_LOG.md`, `report/REPORT_NOTES.md` | Confirmed. `python src/preprocessing.py` created `data/processed/windowed_labeled_data.csv`. | Output was pending until rerun; no raw data, model files, or fake model results added. |
| 2026-06-12 | Codex | Recorded confirmed preprocessing output and class balance. | Preprocessing now runs successfully and produced real processed dataset statistics for the report. | `PROJECT_LOG.md`, `report/REPORT_NOTES.md`, `report/final_report_draft.md` | Confirmed raw shape: 1,516,948 x 17. Confirmed processed shape: 252,720 x 78. Confirmed class balance: 247,520 normal windows and 5,200 failure-risk/anomaly windows. | Dataset is highly imbalanced; emphasize F1-score and recall. No raw or processed CSV files should be committed. |
