# COEN 330 Final Report Draft: MetroPT-3 Binary Classification

## 1. Introduction

This project studies supervised binary classification for the MetroPT-3 air compressor dataset. The goal is to classify sensor records as normal operation or failure-risk/anomaly.

## 2. Dataset

The dataset is the MetroPT-3 Dataset from the UCI Machine Learning Repository. Raw data files are stored locally in `data/raw/` and are not committed to GitHub.

Items to document before final submission:

- Exact raw file used
- Number of rows and columns after loading
- Sensor columns used as features
- Missing-value handling
- Labeling method for normal vs failure-risk/anomaly

## 3. Labeling Strategy

This project requires binary supervised labels. The implementation supports either an existing binary target column or documented timestamp-based failure-risk windows configured in `src/config.py`.

The final report should clearly justify the labeling strategy. Do not report final results until the labels have been verified.

## 4. Methods

Baseline workflow:

- Load local MetroPT-3 CSV data
- Standardize column names
- Create or validate binary labels
- Build numeric features
- Impute missing values using median imputation
- Standardize numeric features
- Train a logistic regression baseline with class balancing

## 5. Evaluation Plan

Planned evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

Because the failure-risk/anomaly class may be rare, recall and F1 score should be emphasized over accuracy alone.

## 6. Results

Final results are not yet reported. Add verified metrics only after running the finalized training and evaluation pipeline.

## 7. Discussion

Discuss:

- Label quality and assumptions
- Class imbalance
- False positives vs false negatives
- Limitations of the baseline model
- Possible improvements with additional models or feature engineering

## 8. Conclusion

This project provides a reproducible supervised learning workflow for MetroPT-3 normal vs failure-risk/anomaly classification. Final conclusions should be based on verified evaluation results.
