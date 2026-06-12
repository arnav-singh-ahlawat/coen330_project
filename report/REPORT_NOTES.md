# Report Notes: Failure-Risk Prediction for a Metro Train Compressor Using Machine Learning

This file collects report-ready explanations for the COEN 330 final report. Metrics listed here are the verified results from the current event-aware pipeline.

## 1. Problem Definition

This project addresses predictive maintenance for a metro train air compressor. The goal is to predict whether a 1-minute compressor sensor window belongs to a failure-risk/anomaly state.

Compressor failures in railway systems matter because they can create high repair cost, service disruption, and operational risk. In a maintenance setting, the value of a model is not only whether it is accurate on average, but whether it can catch windows that indicate elevated failure risk early enough for inspection, maintenance planning, or service intervention.

The machine learning task is supervised binary classification:

- `0` = normal operation
- `1` = failure-risk/anomaly

## 2. Dataset Description

The dataset used for this project is the MetroPT-3 Dataset from the UCI Machine Learning Repository. It contains time-series measurements from an air compressor system, including pressure, temperature, motor current, and other operational sensor readings.

The raw CSV file is stored locally in `data/raw/MetroPT3(AirCompressor).csv`. It is not committed to GitHub because the file is large. The repository should contain code, documentation, report notes, metrics, and plots, while raw data remains local.

Confirmed raw data details after running `python -m src.preprocessing`:

- Raw file: `data/raw/MetroPT3(AirCompressor).csv`
- Raw shape: 1,516,948 rows and 17 columns
- Timestamp column: `timestamp`

## 3. Preprocessing Explanation

The preprocessing workflow loads the raw MetroPT-3 CSV file from `data/raw/`, standardizes column names, detects the timestamp column, and prepares the data for time-window feature engineering.

The confirmed preprocessing command is:

```bash
python -m src.preprocessing
```

This command converts raw timestamped sensor readings into 1-minute windows, applies the configured failure-risk windows, and writes the processed dataset to `data/processed/windowed_labeled_data.csv`.

Confirmed processed output:

- Final shape: 252,720 rows and 78 columns
- File size: about 168 MB
- The processed CSV is a generated local artifact and should not be committed.

## 4. Labeling Explanation

The target variable is binary. A value of `0` represents normal operation, and a value of `1` represents a failure-risk/anomaly condition.

The labeling rule defines failure-risk as the period from one hour before each known failure start time through the failure end time. All other timestamps or windows are labeled normal. This creates a risk-state/early-warning label, not merely a current-failure label. The one-hour lead is included because predictive maintenance is concerned with detecting warning behavior before or during a failure interval, not only identifying records after failure has already begun.

The documented failure periods used for labeling are:

- April 18, 2020 00:00 to April 18, 2020 23:59
- May 29, 2020 23:30 to May 30, 2020 06:00
- June 5, 2020 10:00 to June 7, 2020 14:30
- July 15, 2020 14:30 to July 15, 2020 19:00

In the generated target, each interval is expanded backward by one hour to mark the failure-risk period.

Preprocessing and labeling were verified before threshold tuning. The configured failure-risk windows match the documented failure reports after applying the planned one-hour risk lead. Recomputing `target` from the processed `window_start` column produced 0 mismatches against `data/processed/windowed_labeled_data.csv`, confirming that labels are applied at the 1-minute window level. The final model features exclude both `window_start` and `target`.

## 5. Feature Engineering Explanation

The MetroPT-3 data is time-series data, so raw sensor readings are aggregated into 1-minute windows. Windowing reduces noise, creates consistent examples for classification, and allows the model to learn short-term compressor behavior patterns instead of relying on individual raw sensor rows.

For each numeric sensor column, the preprocessing pipeline computes mean, standard deviation, minimum, maximum, and last value. It also adds `row_count` for each 1-minute window. The resulting one-row-per-minute table is then labeled with the binary target described above.

The confirmed generated feature table contains 252,720 1-minute windows and 78 columns, including engineered sensor statistics, `row_count`, and the binary target label.

## 6. Model Selection Explanation

The final model comparison uses five supervised classification models:

- Logistic Regression baseline
- Decision Tree
- Random Forest
- Extra Trees
- HistGradientBoostingClassifier

Logistic Regression is included as the baseline because it is simple, interpretable, and useful for establishing a minimum expected performance level. Decision Tree, Random Forest, Extra Trees, and HistGradientBoostingClassifier are included because they can model nonlinear relationships between sensor features and failure-risk labels.

Class weighting is used for Logistic Regression, Decision Tree, Random Forest, and Extra Trees because these models support `class_weight="balanced"` and the processed dataset is highly imbalanced. Decision Tree, Random Forest, and HistGradientBoostingClassifier are tuned with small hyperparameter grids. The final model is selected using validation data only, and the held-out test event is evaluated once after model and threshold selection.

## 7. Validation Strategy Explanation

The final validation strategy is an event-aware blocked time split of the processed windowed dataset.

This split is more realistic than random row-level splitting because MetroPT-3 is time-series data. A random split could place nearby correlated windows from the same operating period, or even the same failure event, into both training and testing sets. That would create leakage and overly optimistic performance estimates.

The strict chronological 70/15/15 split was rejected after verification because the final 15% test period contained 37,908 normal windows and 0 failure-risk windows. That made positive-class recall, F2-score, F1-score, and the confusion matrix unsuitable for evaluating a failure-risk prediction model.

The event-aware blocked split uses the configured failure-risk windows to keep whole failure events in only one split. With four known failure events, the first two events are assigned to training, the third event is assigned to validation, and the fourth/final event is assigned to testing. Normal windows before and after each event are included in the same event block so each split contains both normal and failure-risk samples.

The `window_start` column is used only to order windows and assign event blocks. It is not used as a model feature.

Verified split sizes:

- Train: 144,412 windows; class 0 = 142,465, class 1 = 1,947; failure events 1 and 2
- Validation: 30,721 windows; class 0 = 27,798, class 1 = 2,923; failure event 3
- Test: 77,587 windows; class 0 = 77,257, class 1 = 330; failure event 4

## 8. Metrics Explanation

The selected metrics are recall, F2-score, F1-score, confusion matrix, precision, ROC-AUC when possible, and accuracy.

The report should emphasize recall, F2-score, F1-score, and the confusion matrix for class `1`. Accuracy should be reported but not used as the main success metric.

Confirmed class balance in `data/processed/windowed_labeled_data.csv`:

- Class 0 normal: 247,520 windows, 97.94%
- Class 1 failure-risk/anomaly: 5,200 windows, 2.06%

Because the normal class accounts for 97.94% of the processed windows, accuracy alone is not reliable. A model can achieve high accuracy by predicting most windows as normal while still missing the failure-risk windows that matter most.

False negatives are more costly than false positives in this setting. A false positive may trigger extra inspection or operator attention, which has a cost. A false negative means a true failure-risk window is missed, which can allow compressor degradation to continue unnoticed and can contribute to unplanned repair, service disruption, or safety and operational risk.

Recall is emphasized because it measures how many true failure-risk windows the model catches. F2-score is emphasized because it weights recall more heavily than precision, matching the maintenance priority of reducing missed risk windows. F1-score is also reported because it summarizes the balance between precision and recall. The confusion matrix is necessary because it shows the actual false negatives, false positives, true positives, and true negatives rather than hiding them inside a single aggregate score.

Threshold tuning is performed using validation data only. For each model, class-1 probabilities are evaluated across thresholds from 0.01 to 0.99. The workflow records the threshold that maximizes validation F1-score and the threshold that maximizes validation F2-score. The held-out test event is not used for threshold selection.

## 9. Results Summary

Verified validation threshold comparison:

| Model | Default F1 at 0.50 | Best F1 threshold | Best validation F1 | Best F2 threshold | Best validation F2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression baseline | 0.7550 | 0.61 | 0.8216 | 0.61 | 0.8945 |
| Decision Tree | 0.5001 | 0.13 | 0.5001 | 0.13 | 0.3856 |
| Random Forest | 0.5313 | 0.31 | 0.5525 | 0.19 | 0.4790 |
| Extra Trees | 0.0409 | 0.04 | 0.6952 | 0.03 | 0.7717 |
| HistGradientBoostingClassifier | 0.6603 | 0.11 | 0.6652 | 0.11 | 0.5595 |

Selected final model: Logistic Regression baseline, selected using validation-tuned F1-score only. The selected threshold is 0.61. Test data was not used for model or threshold selection.

Verified final event-aware test metrics on held-out failure event 4:

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

At the validation-selected threshold, the final model predicted 113 failure-risk windows, but all were false positives. All 330 true failure-risk windows from held-out event 4 were predicted normal.

The high test accuracy is misleading because the event-4 test block is dominated by normal windows. The recall, F2-score, F1-score, and confusion matrix show that the model failed to detect the held-out failure event despite high accuracy.

## 10. Interpretation of Held-Out Event 4

The event-aware held-out test result demonstrates poor generalization to an unseen failure event. The model performed much better on validation event 3, but it did not identify any positive windows in test event 4. This should be interpreted as a major limitation of supervised learning with only four independent failure events, not as a coding error.

With only four documented failure events, the model has very limited examples of what failure-risk behavior can look like. Training on events 1 and 2 and validating on event 3 may not capture the sensor patterns present in event 4. The final test result therefore shows that the current model cannot be considered deployment-ready.

## 11. Limitations

Current limitations include strong class imbalance, only four known independent failure events, and weak generalization to the held-out final failure event.

The labeling strategy depends on known failure start and end times. If those times are incomplete or inaccurate, the supervised labels may be noisy. The one-hour lead window is a project-defined early-warning interval and should be described as an assumption.

The event-aware split is intentionally strict. It gives a more realistic estimate of unseen-event performance, but it also exposes the difficulty of learning a general failure-risk classifier from such a small number of independent failures.

## 12. Future Work

Future work should focus on improving held-out event recall without tuning on the test set. Useful directions include blocked cross-validation across failure events if more labeled events become available, collecting more failure examples, testing different window sizes, analyzing event-specific feature distributions, and comparing supervised learning with anomaly-detection or semi-supervised approaches.

Any future model improvement should continue to report recall, F2-score, F1-score, and the confusion matrix as primary evidence, with accuracy treated as secondary context.

## 13. Team Contributions

Team contributions are Pending.

This section should be updated with each team member's specific work on data preparation, feature engineering, modeling, evaluation, GitHub/repository management, demo preparation, and report writing.
