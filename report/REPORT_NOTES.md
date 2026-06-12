# Report Notes: Metro Train Compressor Failure Detection Using Machine Learning

This file collects report-ready explanations as the project progresses. Results must remain marked as Pending until metrics are generated and verified.

## 1. Problem Definition

This project addresses predictive maintenance for a metro train air compressor. The goal is to use historical sensor readings to identify whether the system is operating normally or entering a failure-risk/anomaly condition.

The machine learning problem is supervised binary classification. Each time window is assigned one of two classes: normal operation or failure-risk/anomaly. A successful model could help identify compressor behavior that occurs before or during failure periods.

## 2. Dataset Description

The dataset used for this project is the MetroPT-3 Dataset from the UCI Machine Learning Repository. It contains time-series measurements from an air compressor system, including pressure, temperature, motor current, and other operational sensor readings.

The raw CSV file is stored locally in `data/raw/MetroPT3(AirCompressor).csv`. It is not committed to GitHub because the file is large. The repository should contain code, documentation, report notes, metrics, and plots, while raw data remains local.

Confirmed raw data details after running `python src/preprocessing.py`:

- Raw file: `data/raw/MetroPT3(AirCompressor).csv`
- Raw shape: 1,516,948 rows and 17 columns
- Timestamp column: `timestamp`

## 3. Preprocessing Explanation

The preprocessing workflow loads the raw MetroPT-3 CSV file from `data/raw/`, standardizes column names, detects the timestamp column, and prepares the data for time-window feature engineering.

The preprocessing script is designed to print progress messages so that each step is visible when the script runs. If the timestamp column cannot be detected, the script exits with a clear error instead of producing unreliable output.

The confirmed preprocessing command is:

```bash
python src/preprocessing.py
```

This command converts raw timestamped sensor readings into 1-minute windows, applies the configured failure-risk windows, and writes the processed dataset to `data/processed/windowed_labeled_data.csv`.

Confirmed processed output:

- Final shape: 252,720 rows and 78 columns
- File size: about 168 MB
- The processed CSV is a generated local artifact and should not be committed.

## 4. Labeling Explanation

The target variable is binary. A value of `0` represents normal operation, and a value of `1` represents a failure-risk/anomaly condition.

The labeling rule defines failure-risk as the period from one hour before each known failure start time through the failure end time. All other timestamps or windows are labeled normal. This rule is intended to capture both the failure period and the immediate warning period before failure.

The known failure periods used for labeling are April 18, 2020 00:00 to April 18, 2020 23:59; May 29, 2020 23:30 to May 30, 2020 06:00; June 5, 2020 10:00 to June 7, 2020 14:30; and July 15, 2020 14:30 to July 15, 2020 19:00. In the generated target, each of these intervals is expanded backward by one hour to mark the failure-risk period.

## 5. Feature Engineering Explanation

The MetroPT-3 data is time-series data, so raw sensor readings are aggregated into 1-minute windows. Windowing reduces noise, creates consistent examples for classification, and allows the model to learn short-term behavior patterns instead of relying on individual raw sensor rows.

The feature engineering step parses the `timestamp` column as datetime and groups sensor readings into 1-minute windows. For each numeric sensor column, the preprocessing pipeline computes mean, standard deviation, minimum, maximum, and last value. It also adds `row_count` for each 1-minute window. The resulting one-row-per-minute table is then labeled with the binary target described above.

The confirmed generated feature table contains 252,720 1-minute windows and 78 columns, including engineered sensor statistics, `row_count`, and the binary target label.

## 6. Model Selection Explanation

The project will compare multiple supervised classification models. Logistic Regression is included as the baseline because it is simple, interpretable, and useful for establishing a minimum expected performance level.

Decision Tree, Random Forest, Extra Trees, and HistGradientBoostingClassifier are included because they can model nonlinear relationships between sensor features and failure-risk labels. Tree-based models are also commonly effective for tabular feature sets.

## 7. Validation Strategy Explanation

The validation strategy is a chronological time-based split: the earliest 70% of data is used for training, the next 15% for validation, and the final 15% for testing.

This split is used because MetroPT-3 is time-series data. A random split could place nearby timestamps from the same operating period into both training and testing sets, causing data leakage and overly optimistic performance estimates. A time-based split better represents the real use case of training on past data and evaluating on future data.

Implementation status: Pending.

## 8. Metrics Explanation

The selected metrics are accuracy, precision, recall, F1-score, ROC-AUC when possible, and confusion matrix.

The main metric is F1-score because the confirmed processed dataset is highly imbalanced and both false positives and false negatives matter. Recall is the secondary metric because missing true failure-risk windows is especially important in a predictive maintenance setting.

Confirmed class balance in `data/processed/windowed_labeled_data.csv`:

- Class 0 normal: 247,520 windows, 97.94%
- Class 1 failure-risk/anomaly: 5,200 windows, 2.06%

Because the normal class accounts for 97.94% of the processed windows, accuracy alone is not a reliable main metric. A model could achieve high accuracy while still missing many failure-risk windows. The report should emphasize F1-score and recall, especially recall for class 1, while still reporting precision, ROC-AUC when possible, and the confusion matrix.

Metrics are Pending until models are trained and evaluated.

## 9. Results Summary

Model results are Pending.

No model performance should be reported until models are trained and metrics are verified.

Confirmed preprocessing results are available and can be reported:

- Raw input: 1,516,948 rows and 17 columns
- Processed output: 252,720 rows and 78 columns
- Class balance: 247,520 normal windows and 5,200 failure-risk/anomaly windows

## 10. Error Analysis Notes

Error analysis is Pending.

After model evaluation, this section should discuss false positives, false negatives, confusion matrices, class imbalance, and whether errors cluster around specific time periods or operating conditions.

## 11. Limitations

Current limitations include pending chronological split implementation, pending verified model results, and strong class imbalance in the processed dataset.

The labeling strategy depends on known failure start and end times. If those times are incomplete or inaccurate, the supervised labels may be noisy.

## 12. Future Work

Future work includes implementing the chronological train/validation/test split, comparing all selected models, generating verified metrics, and producing final plots for the report.

Additional improvements could include tuning model hyperparameters, analyzing feature importance, testing different window sizes, and improving threshold selection for the failure-risk class.

## 13. Team Contributions

Team contributions are Pending.

This section should be updated with each team member's specific work on data preparation, feature engineering, modeling, evaluation, GitHub/repository management, demo preparation, and report writing.
