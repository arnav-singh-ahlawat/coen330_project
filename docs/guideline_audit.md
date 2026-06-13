# COEN 330 Guideline Audit

Official source of truth: `docs/COEN330_Project_Guidelines.pdf`.

Audit date: 2026-06-13.

Status legend:

- Complete: clear evidence exists in the repository.
- Partial: some evidence exists, but the requirement needs more documentation, verification, or final submission work.
- Missing: no clear evidence was found.

This audit is documentation-only. It does not change training, evaluation, labels, failure windows, or split logic.

## Summary

| Area | Status | Notes |
| --- | --- | --- |
| Problem definition | Complete | The project is clearly framed as supervised binary classification for MetroPT-3 compressor failure-risk prediction. |
| Dataset source and preprocessing | Complete | Dataset source, local raw-data instructions, one-minute windows, labels, and generated outputs are documented. |
| Data quality discussion | Partial | Missing values and outlier-handling limitations are now described in the report draft; licensing/usage constraints should still be reviewed against the dataset source before final submission. |
| EDA | Complete | Required plots and observations exist, and the report draft now discusses class imbalance, event-level positives, selected sensor distributions, missing values, and limited outlier handling. |
| Feature engineering | Complete | One-minute aggregate sensor features are implemented and documented. |
| Five models and baseline | Complete | Five models are implemented, including Logistic Regression baseline. |
| Hyperparameter tuning | Complete | Decision Tree, Random Forest, and HistGradientBoostingClassifier are tuned. |
| Validation strategy | Complete | Event-aware split is implemented and documented; test is not used for model or threshold selection. |
| Evaluation metrics | Partial | Recall, F2, F1, precision, accuracy, ROC-AUC, and confusion matrices exist; PR-AUC is not currently reported. |
| Error analysis and interpretation | Partial | Event-4 false negatives, false positives, and non-deployment-ready status are now discussed; per-window examples remain limited because no separate qualitative example table was generated. |
| Reproducibility | Partial | README, requirements, scripts, seeds, data instructions, results, and demo exist; clean-run verification should be repeated before final submission. |
| Demo | Complete | Command-line demo exists and explicitly states the model is not deployment-ready. |
| Final report | Partial | Draft now follows the 15 required sections and uses real results, but the final PDF is still missing. |
| Team contribution | Partial | Draft now contains placeholders for all five members; actual names and verified individual contributions must be filled in before submission. |
| References and academic integrity | Complete | Draft now includes MetroPT-3, core Python library references, course guideline reference, and source/tool/AI-assistance acknowledgment. |
| Submission package | Missing | No final PDF or `TeamName_Project.zip` submission package is present. |

## Detailed Audit

| Requirement | Status | Evidence | Recommended fixes |
| --- | --- | --- | --- |
| Use a meaningful applied ML problem. | Complete | `README.md`; `report/final_report_draft.md`; `report/REPORT_NOTES.md` | Keep the project framed as predictive maintenance for a metro train compressor. |
| Clearly define the ML task. | Complete | `README.md`; `report/final_report_draft.md` | Keep the task stated as supervised binary classification. |
| Define input features and target/output. | Complete | `src/features.py`; `src/modeling.py`; `report/REPORT_NOTES.md` | In the final report, list major sensor feature groups and explicitly state that `target` and `window_start` are excluded from model features. |
| Define target classes. | Complete | `README.md`; `src/config.py`; `report/final_report_draft.md` | Keep `0 = normal operation` and `1 = failure-risk/anomaly`. |
| Explain motivation, practical value, assumptions, and scope limitations. | Complete | `README.md`; `report/final_report_draft.md`; `report/REPORT_NOTES.md` | Keep the limitation that the current model is not deployment-ready. |
| Document dataset source or collection method. | Complete | `data/data_link.txt`; `README.md`; `report/REPORT_NOTES.md` | Cite the UCI MetroPT-3 source in the final references. |
| Include dataset access and preparation instructions. | Complete | `data/data_link.txt`; `README.md` | Ensure the final README still tells users where to place the raw CSV. |
| Report number of samples and features. | Complete | `report/final_report_draft.md`; `report/REPORT_NOTES.md`; `results/eda_summary.csv` | Keep verified values only: raw 1,516,948 rows x 17 columns; processed 252,720 rows x 78 columns; 76 model features. |
| Describe target labels. | Complete | `src/config.py`; `src/labeling.py`; `report/REPORT_NOTES.md` | Do not change the failure windows unless explicitly instructed. |
| Document data types and time-series structure. | Complete | `README.md`; `report/final_report_draft.md`; `src/features.py` | Keep describing MetroPT-3 as timestamped compressor sensor time-series data. |
| Document missing values and handling. | Complete | `results/eda_summary.csv`; `src/preprocessing.py`; `src/train.py`; `report/REPORT_NOTES.md` | Keep the explanation that standard-deviation features have missing values and model pipelines use median imputation. |
| Document outliers. | Partial | `results/plots/key_sensor_distributions.png`; `report/final_report_draft.md` | The draft now states that boxplot-style distribution EDA exists, numerical outlier counts were not exported, and no outlier-removal rule was applied. A stronger final version could add quantified outlier counts. |
| Document scaling, normalization, encoding, and time-series transformations. | Complete | `src/features.py`; `src/preprocessing.py`; `src/train.py`; `report/REPORT_NOTES.md` | Keep one-minute resampling, aggregate statistics, imputation, and StandardScaler details. |
| Document split or validation strategy. | Complete | `src/modeling.py`; `src/train.py`; `src/evaluate.py`; `README.md`; `report/REPORT_NOTES.md` | Keep the event-aware split as the main realistic evaluation. |
| Discuss data leakage risks and mitigation. | Complete | `README.md`; `report/final_report_draft.md`; `report/REPORT_NOTES.md`; `src/modeling.py` | Keep explaining why random window-level splits are optimistic. |
| Discuss ethical, privacy, licensing, or usage constraints. | Partial | `data/data_link.txt`; `README.md`; `report/final_report_draft.md` | The draft cites UCI and states raw/processed data are not committed. Before final submission, review any dataset-specific license or usage terms from UCI. |
| Include EDA figures. | Complete | `src/eda.py`; `results/plots/class_balance.png`; `results/plots/split_class_balance.png`; `results/plots/sensor_correlation_heatmap.png`; `results/plots/failure_windows_timeline.png`; `results/plots/key_sensor_distributions.png` | Include the most relevant plots in the final report. |
| Include EDA observations. | Complete | `report/final_report_draft.md`; `report/REPORT_NOTES.md` | Keep observations tied to class imbalance, four independent failure events, and split choice. |
| Include missing-value analysis. | Complete | `results/eda_summary.csv`; `report/REPORT_NOTES.md` | Consider turning this into a small table in the report. |
| Include correlations or feature relationships. | Complete | `results/plots/sensor_correlation_heatmap.png`; `src/eda.py`; `report/final_report_draft.md` | Keep the interpretation limited to EDA-selected engineered sensor features. |
| Include representative examples or visualizations. | Partial | `results/plots/failure_windows_timeline.png`; `results/demo_output.txt`; `report/final_report_draft.md` | Draft now includes event-level qualitative examples: event-4 false negatives and false positives. Per-window examples remain unavailable. |
| Explain feature engineering and data representation. | Complete | `src/features.py`; `src/preprocessing.py`; `report/REPORT_NOTES.md` | Keep the one-minute window aggregate-statistics explanation. |
| Justify kept, removed, transformed, or created features. | Complete | `report/REPORT_NOTES.md`; `report/final_report_draft.md`; `src/modeling.py` | Keep `window_start` excluded from model features and used only for ordering/splitting; keep `target` excluded from inputs. |
| Implement at least five models. | Complete | `src/train.py`; `src/stratified_baseline.py`; `results/metrics_table.csv` | Keep the five required models: Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoostingClassifier. |
| Include a simple interpretable baseline. | Complete | `src/train.py`; `report/REPORT_NOTES.md` | Keep Logistic Regression labeled as the baseline. |
| Justify model selection. | Complete | `report/REPORT_NOTES.md`; `report/final_report_draft.md` | Add model-specific justification in the final report if space allows. |
| Tune at least three models. | Complete | `src/train.py`; `results/metrics_table.csv`; `results/threshold_table.csv` | Current tuned models are Decision Tree, Random Forest, and HistGradientBoostingClassifier. Note: the PDF minimum technical requirement says at least three models; the rubric wording could be read broadly, so tune more models only if the instructor expects it. |
| Do not use test set for model or hyperparameter selection. | Complete | `src/train.py`; `src/evaluate.py`; `report/REPORT_NOTES.md` | Keep threshold and model selection validation-only. |
| Use an appropriate validation strategy. | Complete | `src/modeling.py`; `results/training_output.txt`; `report/REPORT_NOTES.md` | Keep train events 1-2, validation event 3, and test event 4. |
| Include model comparison table. | Complete | `results/metrics_table.csv`; `report/final_report_draft.md` | Keep validation and test comparisons separate. |
| Use task-appropriate metrics. | Complete | `src/modeling.py`; `results/metrics_table.csv`; `results/test_metrics.csv` | Continue emphasizing recall, F2, F1, precision, and confusion matrix over accuracy. |
| Include PR-AUC or ROC-AUC where applicable. | Partial | `src/modeling.py`; `results/metrics_table.csv`; `results/test_metrics.csv`; `results/plots/precision_recall_curve.png` | ROC-AUC is reported. Consider adding PR-AUC because the PDF lists it for imbalanced classification. |
| Discuss model complexity, interpretability, and computational cost. | Partial | `report/REPORT_NOTES.md`; `report/final_report_draft.md` | Draft now distinguishes the interpretable Logistic Regression baseline from nonlinear tree/ensemble models; computational-cost discussion remains brief. |
| Perform error analysis. | Partial | `report/REPORT_NOTES.md`; `report/final_report_draft.md`; `results/test_metrics.csv`; `results/plots/confusion_matrix.png` | Draft now includes false-positive and false-negative discussion. Per-window examples remain unavailable. |
| Interpret important features. | Partial | `results/eda_output.txt`; `results/plots/sensor_correlation_heatmap.png`; `results/plots/key_sensor_distributions.png`; `report/final_report_draft.md` | Draft now discusses EDA-selected engineered sensor features and honestly states no reliable final feature-importance table was exported. |
| Discuss limitations and future work. | Complete | `report/final_report_draft.md`; `report/REPORT_NOTES.md` | Keep the limitation that held-out event 4 had recall, F2, and F1 of 0.0. |
| Provide clear folder structure. | Complete | `README.md` | Keep folder structure updated if files move. |
| Provide README run instructions. | Complete | `README.md` | Before final submission, rerun commands from a clean shell and update if anything changes. |
| Provide dependencies. | Complete | `requirements.txt` | Consider pinning versions if exact reproducibility is required. |
| Provide preprocessing, training, validation, evaluation, and visualization scripts. | Complete | `src/preprocessing.py`; `src/train.py`; `src/evaluate.py`; `src/eda.py`; `src/stratified_baseline.py` | Keep scripts runnable as modules from the project root. |
| Provide dataset link or instructions. | Complete | `data/data_link.txt`; `README.md` | Keep raw data local and uncommitted. |
| Include saved results and figures where appropriate. | Complete | `results/metrics_table.csv`; `results/test_metrics.csv`; `results/threshold_table.csv`; `results/plots/` | Do not replace verified results unless the pipeline is rerun intentionally. |
| Use clear random seeds. | Complete | `src/config.py`; `src/train.py`; `src/stratified_baseline.py` | Keep `RANDOM_STATE = 42` documented. |
| Exclude raw/processed data and large models from commits. | Complete | `.gitignore`; `data/data_link.txt`; `README.md` | Local raw and processed CSV files and `.joblib` models exist, but `.gitignore` excludes them. Do not force-add them. |
| Include a minimal demo. | Complete | `demo/demo.py`; `demo/README.md`; `results/demo_output.txt` | Optional: add a screenshot or short video if requested. |
| Demo uses valid model features. | Complete | `demo/demo.py`; `results/demo_output.txt` | Keep using processed one-minute window features, not raw-style hand-built features. |
| Demo avoids deployment-ready claim. | Complete | `demo/demo.py`; `demo/README.md`; `README.md` | Keep the explicit non-deployment-ready warning. |
| Final report has required 15 sections. | Complete | `report/final_report_draft.md` | Draft is now organized into the 15 PDF-required sections. Review content and convert to PDF before submission. |
| Final report submitted as PDF. | Missing | No `report/final_report.pdf` found. | Generate a final PDF after the draft is complete and reviewed. |
| Team contribution section. | Partial | `report/final_report_draft.md`; `report/REPORT_NOTES.md` | Placeholders for five members were added. Replace placeholders with final names and concrete contributions before submission. |
| References. | Complete | `report/final_report_draft.md` | References now include MetroPT-3/UCI, scikit-learn, pandas, NumPy, matplotlib, seaborn, and the course guideline PDF. |
| Academic integrity and external-tool acknowledgment. | Complete | `report/final_report_draft.md` | Draft now acknowledges dataset source, Python libraries, and AI-assisted tools for coding/report organization/debugging. |
| Submit one Moodle ZIP package. | Missing | No `TeamName_Project.zip` found. | Prepare the final package only after report PDF, source, README, dataset instructions, results, figures, and demo files are final. |
| Presentation and Q&A if requested. | Missing | No presentation files found. | Create slides only if the instructor requests a presentation. If no presentation is required, the PDF says that weight may be redistributed to final report quality. |

## Current Evidence Of Main Results

- Event-aware main split: train failure events 1-2, validation failure event 3, test failure event 4. Evidence: `src/modeling.py`, `results/training_output.txt`, `report/REPORT_NOTES.md`.
- Final selected model: Logistic Regression baseline, selected by validation-tuned F1 threshold. Evidence: `results/threshold_table.csv`, `results/test_metrics.csv`.
- Final event-aware held-out event-4 result: accuracy `0.9942902805882429`, precision `0.0`, recall `0.0`, F1 `0.0`, F2 `0.0`, ROC-AUC `0.12087550368094527`, confusion matrix `[[77144, 113], [330, 0]]`. Evidence: `results/test_metrics.csv`.
- Important interpretation: the event-aware test failed to generalize to held-out failure event 4. This must remain visible in the report.
- Stratified baseline: optimistic window-level comparison only, not deployment performance. Evidence: `src/stratified_baseline.py`, `results/stratified_baseline_metrics.csv`, `report/REPORT_NOTES.md`.

## Priority Fix List

1. Replace team contribution placeholders with final names and verified individual contributions.
2. Generate `report/final_report.pdf`.
3. Rerun clean reproduction commands before claiming completion.
4. Build the final `TeamName_Project.zip` Moodle submission package without raw data, processed CSV files, virtual environments, or large model files.
5. Consider adding PR-AUC for the imbalanced classification evaluation only if metrics are intentionally regenerated.
6. Consider adding quantified outlier counts or per-window qualitative examples only if generated from existing data without changing labels, failure windows, split logic, model selection, or reported metrics.
