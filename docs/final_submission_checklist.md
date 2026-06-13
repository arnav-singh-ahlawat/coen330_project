# Final Submission Checklist

Project: Failure-Risk Prediction for a Metro Train Compressor Using Machine Learning

Checklist date: 2026-06-13.

This checklist is for the final Moodle ZIP package. It does not change training, evaluation, labels, failure windows, event-aware split logic, model selection logic, thresholds, or reported metrics.

## Ready

- `report/final_report.md` exists and is organized into the 15 required report sections.
- `report/final_report.pdf` exists and was generated locally from `report/final_report.md`.
- The final report states that the task is supervised binary classification with `0 = normal operation` and `1 = failure-risk/anomaly`.
- The final report states the failure-risk label definition: 1 hour before documented failure start through documented failure end.
- The final report states that 5,200 positive windows are not 5,200 independent failures and come from only 4 independent failure events.
- The final report keeps the event-aware blocked split as the main realistic evaluation: train events 1-2, validation event 3, test event 4.
- The final report states that the selected Logistic Regression baseline at threshold 0.61 failed to generalize to held-out event 4: TP = 0, FN = 330, recall = 0.0, F1 = 0.0, F2 = 0.0, ROC-AUC = 0.1209.
- The final report states that high test accuracy is misleading because the test set is dominated by normal windows.
- The final report states that the model is not deployment-ready.
- The final report treats the stratified window-level baseline as optimistic only.
- Required result tables exist in `results/`: `metrics_table.csv`, `threshold_table.csv`, `test_metrics.csv`, `stratified_baseline_metrics.csv`, `eda_summary.csv`, and `event_summary.csv`.
- Required plots exist in `results/plots/`: `class_balance.png`, `split_class_balance.png`, `sensor_correlation_heatmap.png`, `failure_windows_timeline.png`, `key_sensor_distributions.png`, `confusion_matrix.png`, `precision_recall_curve.png`, `validation_probability_distribution.png`, `test_probability_distribution.png`, and `stratified_baseline_confusion_matrix.png`.
- Core source scripts exist in `src/`, including preprocessing, EDA, training, evaluation, modeling helpers, and the stratified baseline.
- Demo script exists in `demo/demo.py`.
- Dataset source/instructions exist in `data/data_link.txt`.
- `README.md`, `requirements.txt`, `PROJECT_LOG.md`, and `report/REPORT_NOTES.md` exist.
- Final verification command `git diff --check` passed on 2026-06-13.
- Final verification command `python demo/demo.py` passed on 2026-06-13.

## Missing or Manual Before Moodle Submission

- Replace `TODO_NAME` placeholders in `report/final_report.md` and regenerate `report/final_report.pdf` if exact team member names are available.
- Build the final Moodle ZIP package. No final `TeamName_Project.zip` has been created by this checklist.
- Review any dataset-specific UCI license or usage terms before public redistribution.
- Decide whether to include `models/final_model.joblib`. It is small locally and useful for the demo, but model artifacts are ignored by Git and should only be included intentionally in the ZIP.
- Add a demo screenshot or short video link only if the instructor requests or expects one.

## Include in Final Moodle ZIP

- `README.md`
- `requirements.txt`
- `PROJECT_LOG.md`
- `data/data_link.txt`
- `src/`
- `demo/`
- `docs/`
- `results/`
- `results/plots/`
- `report/final_report.md`
- `report/final_report.pdf`
- Consider including `models/final_model.joblib` if it remains small and the team wants the demo to run immediately without retraining.

## Exclude from Final Moodle ZIP

- `.git/`
- `.venv/`
- `__pycache__/` and `**/__pycache__/`
- `data/raw/`
- `data/processed/windowed_labeled_data.csv`
- Other generated processed CSV files in `data/processed/`, unless the instructor explicitly requires them.
- Raw dataset ZIP files or extracted raw data files.
- Large temporary files.
- Large model files or experiment artifacts not needed for review.

## Final Packaging Reminder

Use a ZIP name matching the course guideline format:

```text
TeamName_Project.zip
```

Before claiming the package is complete, verify that the report PDF opens, the demo works, and the ZIP does not contain raw data, processed CSV files, a virtual environment, or Git history.
