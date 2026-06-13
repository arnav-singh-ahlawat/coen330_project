# Failure-Risk Prediction for a Metro Train Compressor Using Machine Learning

## 1. Introduction

This project studies predictive maintenance for a metro train air compressor using the MetroPT-3 dataset. The goal is to predict whether a 1-minute compressor sensor window belongs to a failure-risk/anomaly state.

Compressor failures in railway systems can cause high repair cost, service disruption, and operational risk. For that reason, the model should prioritize catching failure-risk windows over maximizing overall accuracy. A missed failure-risk window is more serious than an unnecessary warning because it can mean a missed opportunity to inspect or repair the compressor before a failure affects service.

The machine learning task is supervised binary classification:

- `0` = normal operation
- `1` = failure-risk/anomaly

Project decision history, reproducibility notes, commands run, problems encountered, and fixes applied are tracked in `PROJECT_LOG.md`. Additional report-ready explanations are collected in `report/REPORT_NOTES.md`.

## 2. Dataset

The dataset is the MetroPT-3 Dataset from the UCI Machine Learning Repository. It contains real time-series sensor readings from an air compressor system, including pressure, temperature, motor current, and other operational measurements.

The confirmed raw file used for this project is `data/raw/MetroPT3(AirCompressor).csv`. After running `python -m src.preprocessing`, the raw data was confirmed to contain 1,516,948 rows and 17 columns. The timestamp column is named `timestamp`.

The generated processed file is `data/processed/windowed_labeled_data.csv`. It contains 252,720 rows and 78 columns and is about 168 MB. This processed CSV is a local generated artifact and should not be committed to GitHub.

The confirmed class balance in the processed dataset is:

- Class 0 normal: 247,520 windows, 97.94%
- Class 1 failure-risk/anomaly: 5,200 windows, 2.06%

This class balance shows that the dataset is highly imbalanced. A high accuracy score can be misleading because most windows are normal.

## 3. Labeling Strategy

The target label is created from documented failure periods. A 1-minute window is labeled failure-risk if its timestamp occurs from 1 hour before a documented failure start until the documented failure end. All other windows are labeled normal.

This is a risk-state/early-warning label, not merely a current-failure label. The one-hour lead time is included because predictive maintenance is concerned with detecting warning behavior before or during a failure interval, not only identifying records after failure has already started.

The labeling strategy produced 5,200 failure-risk/anomaly windows in the confirmed processed dataset. Labeling was verified by recomputing the target from `window_start` and the configured failure-risk windows; the recomputed labels had 0 mismatches with the saved processed dataset.

## 4. Methods

Preprocessing and feature engineering workflow:

- Load local MetroPT-3 CSV data
- Standardize column names
- Detect and parse the `timestamp` column
- Convert raw timestamped sensor readings into 1-minute windows
- For each numeric sensor column, compute mean, standard deviation, minimum, maximum, and last value
- Add `row_count` for each 1-minute window
- Add the binary target label using the configured failure-risk windows

The confirmed processed dataset has 252,720 one-minute windows and 78 columns. Model training and evaluation use this windowed, labeled dataset rather than the raw timestamp-level records.

The model comparison includes Logistic Regression, Decision Tree, Random Forest, Extra Trees, and HistGradientBoostingClassifier. Logistic Regression is used as the baseline. Class weighting is used where supported because the positive failure-risk class is rare.

## 4.1 Exploratory Data Analysis

Exploratory data analysis was generated with `src/eda.py` using `data/processed/windowed_labeled_data.csv`. The script created class-balance, split-balance, sensor-correlation, timeline, and class-wise sensor-distribution plots, along with `results/eda_summary.csv` and `results/event_summary.csv`.

The EDA confirms that the dataset is highly imbalanced. The processed data contains 247,520 normal windows and 5,200 failure-risk windows, so only 2.06% of windows are positive. This imbalance explains why accuracy is misleading: a model can predict most windows as normal and still appear accurate while failing to detect the rare failure-risk class.

The event summary also shows that the 5,200 positive windows come from only four independent failure events. These are not 5,200 independent failures; they are repeated 1-minute windows around four documented compressor failure periods. This supports the decision to evaluate by event rather than by a random window-level split.

The EDA supports using recall, F2-score, F1-score, and the confusion matrix as primary evaluation evidence. Recall measures how many true failure-risk windows are caught, F2-score weights recall more heavily, F1-score summarizes the precision/recall tradeoff, and the confusion matrix shows the actual false negatives and true positives that accuracy can hide.

The timeline plot shows the four separated clusters of failure-risk windows over time. This visual pattern supports the event-aware split: events 1 and 2 are used for training, event 3 for validation, and event 4 for testing.

## 5. Validation Strategy

The final evaluation uses an event-aware blocked split:

- Training: failure events 1 and 2 plus surrounding normal windows
- Validation: failure event 3 plus surrounding normal windows
- Test: failure event 4 plus surrounding normal windows

This validation strategy is more realistic than random row-level splitting. MetroPT-3 is time-series data, so nearby windows can be strongly correlated. A random split could place windows from the same operating period or the same failure event into both training and testing sets, causing leakage and overly optimistic performance.

The event-aware split keeps each failure event entirely within one split. The held-out test set therefore asks whether patterns learned from earlier failure events generalize to a later unseen failure event.

The verified split sizes are:

- Train: 144,412 windows; class 0 = 142,465, class 1 = 1,947
- Validation: 30,721 windows; class 0 = 27,798, class 1 = 2,923
- Test: 77,587 windows; class 0 = 77,257, class 1 = 330

A secondary stratified window-level baseline is included only as an optimistic comparison. It uses a stratified random split over processed windows, so it can mix windows from the same failure events across train and test. This baseline checks whether the engineered features can separate labeled risk windows when event independence is not enforced, but it should not be interpreted as deployment performance.

## 6. Evaluation Metrics

The project reports recall, F2-score, F1-score, the confusion matrix, precision, ROC-AUC when possible, and accuracy.

The main evaluation emphasis is recall, F2-score, F1-score, and the confusion matrix for the failure-risk class. Accuracy is reported but should not be used as the main success metric because the dataset is highly imbalanced.

False negatives are more costly than false positives in railway predictive maintenance. A false positive may trigger extra inspection, but a false negative misses a true failure-risk window and can allow compressor degradation to continue unnoticed. Recall is emphasized because it measures how many true failure-risk windows are caught. F2-score is emphasized because it gives more weight to recall than precision. F1-score is also useful because it summarizes the precision/recall tradeoff. The confusion matrix is necessary because it shows the actual numbers of false negatives and true positives.

Threshold tuning is performed on the validation set only. The test set is not used for model selection or threshold selection.

## 7. Results

The selected final model is the Logistic Regression baseline with threshold 0.61, selected using validation-tuned F1-score. The same threshold also produced the best validation F2-score for this model. The held-out test event was evaluated once after model and threshold selection.

Verified validation threshold comparison:

| Model | Default F1 at 0.50 | Best F1 threshold | Best validation F1 | Best F2 threshold | Best validation F2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression baseline | 0.7550 | 0.61 | 0.8216 | 0.61 | 0.8945 |
| Decision Tree | 0.5001 | 0.13 | 0.5001 | 0.13 | 0.3856 |
| Random Forest | 0.5313 | 0.31 | 0.5525 | 0.19 | 0.4790 |
| Extra Trees | 0.0409 | 0.04 | 0.6952 | 0.03 | 0.7717 |
| HistGradientBoostingClassifier | 0.6603 | 0.11 | 0.6652 | 0.11 | 0.5595 |

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

The high test accuracy is misleading because the held-out test block contains many more normal windows than failure-risk windows. The more important metrics show that the model did not detect any of the 330 true failure-risk windows in event 4.

The secondary stratified window-level baseline produced much stronger results:

| Model | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression baseline | 0.9636 | 0.3573 | 0.9615 | 0.5210 | 0.7185 | 0.9849 |
| Decision Tree | 0.9951 | 0.8421 | 0.9385 | 0.8877 | 0.9175 | 0.9687 |
| Random Forest | 0.9963 | 0.8638 | 0.9760 | 0.9165 | 0.9513 | 0.9950 |
| Extra Trees | 0.9963 | 0.8695 | 0.9673 | 0.9158 | 0.9460 | 0.9926 |
| HistGradientBoostingClassifier | 0.9955 | 0.8301 | 0.9817 | 0.8996 | 0.9471 | 0.9986 |

These results come from `src/stratified_baseline.py`, using a stratified train/test split with 202,176 train windows and 50,544 test windows. They are included to show how the evaluation can look when windows are randomly mixed, not to replace the event-aware test result.

## 8. Discussion

The final event-aware test result demonstrates poor generalization to an unseen failure event. The model achieved strong validation performance on event 3, but it failed to identify any positive windows in held-out event 4. This should be presented as a major limitation of supervised learning with only four independent failure events.

This finding should not be hidden or treated as a coding error. The preprocessing, labeling, and event-aware split were verified. The result means that the patterns learned from events 1 and 2, with threshold selection on event 3, did not transfer to event 4.

The dataset limitation is important: four documented failures provide very few independent examples of failure-risk behavior. A supervised model may learn event-specific patterns that do not generalize. More labeled failure events would be needed to better estimate generalization and improve reliability.

The stratified baseline reinforces this point. The processed dataset contains 5,200 positive windows, but those windows come from only four independent failure events. A random or stratified window-level split can place similar windows from the same event into both training and testing, which makes performance look much stronger than the event-aware held-out result. This comparison demonstrates why random window-level splits can be misleading for predictive maintenance.

## 9. Conclusion

This project provides a reproducible supervised learning workflow for failure-risk prediction in a metro train compressor using the MetroPT-3 dataset. The project keeps the raw dataset local, creates verified 1-minute labeled windows, uses an event-aware validation strategy, and reports metrics that reflect predictive maintenance priorities.

The current final model is not deployment-ready. Although test accuracy is high, recall, F2-score, and F1-score for the failure-risk class are all 0.0 on the held-out failure event. The main conclusion is that event-aware validation exposed poor generalization to unseen failure event 4, highlighting the need for more independent failure events and further modeling work.

The stratified baseline is useful as an optimistic feature-separability check, but the event-aware evaluation remains the realistic result for deployment.
