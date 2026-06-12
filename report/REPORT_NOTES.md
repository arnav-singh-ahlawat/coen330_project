# Report Notes: Metro Train Compressor Failure Detection Using Machine Learning

This file collects report-ready explanations as the project progresses. Results must remain marked as Pending until metrics are generated and verified.

## 1. Problem Definition

This project addresses predictive maintenance for a metro train air compressor. The goal is to use historical sensor readings to identify whether the system is operating normally or entering a failure-risk/anomaly condition.

The machine learning problem is supervised binary classification. Each time window is assigned one of two classes: normal operation or failure-risk/anomaly. A successful model could help identify compressor behavior that occurs before or during failure periods.

## 2. Dataset Description

The dataset used for this project is the MetroPT-3 Dataset from the UCI Machine Learning Repository. It contains time-series measurements from an air compressor system, including pressure, temperature, motor current, and other operational sensor readings.

The raw CSV file is stored locally in `data/raw/MetroPT3(AirCompressor).csv`. It is not committed to GitHub because the file is large. The repository should contain code, documentation, report notes, metrics, and plots, while raw data remains local.

## 3. Preprocessing Explanation

The preprocessing workflow loads the raw MetroPT-3 CSV file from `data/raw/`, standardizes column names, detects the timestamp column, and prepares the data for time-window feature engineering.

The preprocessing script is designed to print progress messages so that each step is visible when the script runs. If the timestamp column cannot be detected, the script exits with a clear error instead of producing unreliable output.

## 4. Labeling Explanation

The target variable is binary. A value of `0` represents normal operation, and a value of `1` represents a failure-risk/anomaly condition.

The planned labeling rule defines failure-risk as the period from one hour before each known failure start time through the failure end time. All other timestamps or windows are labeled normal. This rule is intended to capture both the failure period and the immediate warning period before failure.

Label generation is Pending until known failure windows are configured in `src/config.py`.

## 5. Feature Engineering Explanation

The MetroPT-3 data is time-series data, so raw sensor readings are aggregated into 1-minute windows. Windowing reduces noise, creates consistent examples for classification, and allows the model to learn short-term behavior patterns instead of relying on individual raw sensor rows.

The feature engineering plan is to compute mean, standard deviation, minimum, maximum, and last value for each numeric sensor in each 1-minute window. Current implementation includes mean, standard deviation, minimum, maximum, last value, and row count.

## 6. Model Selection Explanation

The project will compare multiple supervised classification models. Logistic Regression is included as the baseline because it is simple, interpretable, and useful for establishing a minimum expected performance level.

Decision Tree, Random Forest, Extra Trees, and HistGradientBoostingClassifier are included because they can model nonlinear relationships between sensor features and failure-risk labels. Tree-based models are also commonly effective for tabular feature sets.

## 7. Validation Strategy Explanation

The validation strategy is a chronological time-based split: the earliest 70% of data is used for training, the next 15% for validation, and the final 15% for testing.

This split is used because MetroPT-3 is time-series data. A random split could place nearby timestamps from the same operating period into both training and testing sets, causing data leakage and overly optimistic performance estimates. A time-based split better represents the real use case of training on past data and evaluating on future data.

Implementation status: Pending.

## 8. Metrics Explanation

The selected metrics are accuracy, precision, recall, F1-score, ROC-AUC when possible, and confusion matrix.

The main metric is F1-score because the failure-risk/anomaly class may be imbalanced and both false positives and false negatives matter. Recall is the secondary metric because missing true failure-risk windows is especially important in a predictive maintenance setting.

Metrics are Pending until models are trained and evaluated.

## 9. Results Summary

Results are Pending.

No model performance should be reported until the labeling windows are configured, preprocessing successfully generates the labeled window dataset, models are trained, and metrics are verified.

## 10. Error Analysis Notes

Error analysis is Pending.

After model evaluation, this section should discuss false positives, false negatives, confusion matrices, class imbalance, and whether errors cluster around specific time periods or operating conditions.

## 11. Limitations

Current limitations include missing configured failure windows, pending final labeled dataset generation, pending chronological split implementation, and pending verified model results.

The labeling strategy depends on known failure start and end times. If those times are incomplete or inaccurate, the supervised labels may be noisy.

## 12. Future Work

Future work includes configuring known failure windows, completing the 1-minute feature engineering plan, implementing the chronological train/validation/test split, comparing all selected models, generating verified metrics, and producing final plots for the report.

Additional improvements could include tuning model hyperparameters, analyzing feature importance, testing different window sizes, and improving threshold selection for the failure-risk class.

## 13. Team Contributions

Team contributions are Pending.

This section should be updated with each team member's specific work on data preparation, feature engineering, modeling, evaluation, GitHub/repository management, demo preparation, and report writing.
