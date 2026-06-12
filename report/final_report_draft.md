# COEN 330 Final Report Draft: MetroPT-3 Binary Classification

## 1. Introduction

This project studies supervised binary classification for the MetroPT-3 air compressor dataset. The goal is to classify sensor records as normal operation or failure-risk/anomaly.

Project decision history, reproducibility notes, commands run, problems encountered, and fixes applied are tracked in `PROJECT_LOG.md`. Report-ready explanations are collected in `report/REPORT_NOTES.md`.

## 2. Dataset

The dataset is the MetroPT-3 Dataset from the UCI Machine Learning Repository. Raw data files are stored locally in `data/raw/` and are not committed to GitHub.

The confirmed raw file used for this project is `data/raw/MetroPT3(AirCompressor).csv`. After running `python src/preprocessing.py`, the raw data was confirmed to contain 1,516,948 rows and 17 columns. The timestamp column is named `timestamp`.

The generated processed file is `data/processed/windowed_labeled_data.csv`. It contains 252,720 rows and 78 columns and is about 168 MB. This processed CSV is a local generated artifact and should not be committed to GitHub.

The confirmed class balance in the processed dataset is:

- Class 0 normal: 247,520 windows, 97.94%
- Class 1 failure-risk/anomaly: 5,200 windows, 2.06%

## 3. Labeling Strategy

This project requires binary supervised labels. The implementation supports either an existing binary target column or documented timestamp-based failure-risk windows configured in `src/config.py`.

Failure-risk is defined as one hour before each known failure start time through the failure end time. All other windows are labeled normal. This labeling strategy produced 5,200 failure-risk/anomaly windows in the confirmed processed dataset.

## 4. Methods

Preprocessing and feature engineering workflow:

- Load local MetroPT-3 CSV data
- Standardize column names
- Detect and parse the `timestamp` column
- Convert raw timestamped sensor readings into 1-minute windows
- For each numeric sensor column, compute mean, standard deviation, minimum, maximum, and last value
- Add `row_count` for each 1-minute window
- Add the binary target label using the configured failure-risk windows

The confirmed processed dataset has 252,720 one-minute windows and 78 columns. Model training and evaluation should use this windowed, labeled dataset rather than the raw timestamp-level records.

## 5. Evaluation Plan

Planned evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC when possible
- Confusion matrix

Because the processed dataset is highly imbalanced, accuracy alone is not a reliable main metric. The normal class represents 97.94% of windows, while the failure-risk/anomaly class represents 2.06%. F1-score and recall should be emphasized, especially recall for class 1, because missing failure-risk windows is costly in a predictive maintenance setting.

## 6. Results

Final model results are not yet reported. Add verified model metrics only after running the finalized training and evaluation pipeline.

Confirmed preprocessing results that can be reported:

- Raw input: 1,516,948 rows and 17 columns
- Processed output: 252,720 rows and 78 columns
- Class balance: 247,520 normal windows and 5,200 failure-risk/anomaly windows

## 7. Discussion

Discuss:

- Label quality and assumptions
- Class imbalance: 97.94% normal and 2.06% failure-risk/anomaly
- False positives vs false negatives
- Limitations of the baseline model
- Possible improvements with additional models or feature engineering

## 8. Conclusion

This project provides a reproducible supervised learning workflow for MetroPT-3 normal vs failure-risk/anomaly classification. Final conclusions should be based on verified evaluation results.
