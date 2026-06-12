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

## 3. Machine Learning Task

Task type: supervised binary classification.

The model should classify each timestamp/window as either normal operation or failure-risk/anomaly condition.

## 4. Target Label Definition

Target column:

- `0` = normal operation
- `1` = failure-risk/anomaly condition

Labeling rule: failure-risk is defined as one hour before each known failure start time until the failure end time. All other timestamps/windows are labeled normal.

Status: Pending until known failure windows are configured in `src/config.py`.

## 5. Feature Engineering Plan

Raw time-series sensor readings are aggregated into 1-minute windows.

For each numeric sensor, compute:

- Mean
- Standard deviation
- Minimum
- Maximum
- Last value

Current implementation status: 1-minute windows with mean, standard deviation, minimum, maximum, last value, and row count have been implemented.

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

Reason: failure-risk/anomaly classification may be imbalanced. Accuracy alone can be misleading if the normal class is much larger than the failure-risk class.

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

Result: `python -m compileall src` passed. `python src/preprocessing.py` exited with a clear error because failure windows are not yet configured.

## 11. Problems Encountered

- `python src/preprocessing.py` originally ran with no useful output and did not clearly show whether `data/processed/windowed_labeled_data.csv` was created.
- `src/preprocessing.py` originally only defined an sklearn numeric preprocessing pipeline and did not run the data preparation workflow.
- `FAILURE_RISK_WINDOWS` is currently empty in `src/config.py`, so valid supervised labels cannot yet be generated from failure windows.
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

## 13. Results Generated

Model metrics: Pending.

Plots: Pending.

Processed windowed labeled dataset: Pending because `FAILURE_RISK_WINDOWS` is not yet configured.

Confirmed current behavior: preprocessing does not create fake labels or fake results. It exits with a clear missing-failure-windows error.

## 14. Report Notes

The final report should explain:

- Why MetroPT-3 is appropriate for predictive maintenance.
- Why the task is supervised binary classification.
- How labels are defined from known failure windows.
- Why 1-minute windows are used.
- Why time-based splitting is used instead of random splitting.
- Why F1-score and recall matter for the failure-risk class.
- Which results are verified and which are pending.

Detailed report-ready notes are collected in `report/REPORT_NOTES.md`.

## 15. Next Steps

- Add the known MetroPT-3 failure windows to `src/config.py`.
- Add last-value aggregation to the 1-minute feature engineering function.
- Run `python src/preprocessing.py` and confirm `data/processed/windowed_labeled_data.csv` is created.
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
