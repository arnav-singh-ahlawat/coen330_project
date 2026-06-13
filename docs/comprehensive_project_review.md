# Comprehensive Project Review

Project: Failure-Risk Prediction for a Metro Train Compressor Using Machine Learning

Review purpose: run the COEN 330 project from start to finish, verify required outputs,
and document the current state without changing labels, failure windows, split logic,
model selection logic, thresholds, or reported metrics.

Source documents used:

- `docs/COEN330_Project_Guidelines.pdf`
- `AGENTS.md`
- `docs/guideline_checklist.md`
- `docs/guideline_audit.md`
- `PROJECT_LOG.md`
- `README.md`
- `report/REPORT_NOTES.md`
- `report/final_report_draft.md`

## Overall Result

The full pipeline passed from the project root. The combined run log is saved in
`results/full_reproducibility_run.txt`.

The event-aware evaluation remains the main realistic result. It still failed to
generalize to held-out failure event 4: the selected model had precision, recall,
F1-score, and F2-score of `0.0`, with 330 false negatives and 0 true positives.
This is not deployment-ready performance and must remain visible in the report.

No ML methodology was changed during this review.

## Step 1 - Repository State Check

Repository state at the start of the review:

| Item | Value |
| --- | --- |
| Current branch | `main` |
| Latest commit hash | `4ebf6addcc1ec6e618a2f04dcd3d6c3bf3032f03` |
| Starting git status | Clean; `git status --short` printed no entries |
| Python version | `Python 3.14.4` |
| Requirements file | `requirements.txt`, 60 bytes |
| Requirements contents | `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `jupyter` |

`python -m pip freeze` ran successfully but printed a cache warning because
`/home/niraj/.cache/pip` is not writable. Key installed package versions observed:

| Package | Installed version |
| --- | --- |
| pandas | `3.0.3` |
| numpy | `2.4.6` |
| scikit-learn | `1.9.0` |
| matplotlib | `3.11.0` |
| seaborn | `0.13.2` |
| joblib | `1.5.3` |
| jupyter | `1.1.1` |

Ignore-rule check:

| Artifact class | Ignored? | Evidence |
| --- | --- | --- |
| `.venv` | Yes | `.gitignore:4:.venv/` |
| Raw data | Yes | `.gitignore:11:data/raw/` |
| Processed CSV data | Yes | `.gitignore:17:data/processed/*.csv` |
| Model files | Yes | `.gitignore:25:models/*.joblib` |

Local generated/data artifact check:

| Path | Exists locally? | Size |
| --- | ---: | ---: |
| `.venv` | Yes | 92 bytes, directory entry size |
| `data/raw/MetroPT3(AirCompressor).csv` | Yes | 218,300,507 bytes |
| `data/processed/windowed_labeled_data.csv` | Yes | 175,958,810 bytes |
| `models/final_model.joblib` | Yes | 6,097 bytes |

Tracked data/model check: `git ls-files data/raw data/processed models .venv` only
returned `data/processed/.gitkeep`. Raw data, processed CSV files, virtual
environments, and model files are not tracked.

Repository state after the full run and log updates:

- `PROJECT_LOG.md` modified by this review.
- `report/REPORT_NOTES.md` modified by this review.
- `results/metrics_table.csv` modified by the rerun only in last-decimal Random Forest
  probability values.
- `results/threshold_table.csv` modified by the rerun only in last-decimal Random Forest
  probability values.
- `results/full_reproducibility_run.txt` created.
- `docs/comprehensive_project_review.md` created.

## Step 2 - Full Pipeline Run

The following commands were run from `/home/niraj/Documents/MetroPT3_Project` and
captured in `results/full_reproducibility_run.txt`:

```bash
python -m src.preprocessing
python -m src.eda
python -m src.train
python -m src.evaluate
python -m src.stratified_baseline
python demo/demo.py
```

Run metadata:

| Item | Value |
| --- | --- |
| Log file | `results/full_reproducibility_run.txt` |
| Log size | 15,514 bytes |
| Log lines | 235 |
| Started | `2026-06-12T23:36:39-04:00` |
| Completed | `2026-06-12T23:38:33-04:00` |

Command results:

| Command | Exit code | Result |
| --- | ---: | --- |
| `python -m src.preprocessing` | 0 | Passed; regenerated `data/processed/windowed_labeled_data.csv` |
| `python -m src.eda` | 0 | Passed; regenerated EDA summaries and plots |
| `python -m src.train` | 0 | Passed; trained five models, tuned validation thresholds, saved final model and result tables |
| `python -m src.evaluate` | 0 | Passed; evaluated held-out event 4 with selected validation threshold |
| `python -m src.stratified_baseline` | 0 | Passed; regenerated optimistic stratified baseline metrics and plot |
| `python demo/demo.py` | 0 | Passed; loaded model, selected threshold, and printed representative predictions |

No command failed, so the failure-stop condition was not triggered.

## Step 3 - Required Output Files

Core files:

| Path | Status | Size |
| --- | --- | ---: |
| `README.md` | Exists | 3,780 bytes |
| `requirements.txt` | Exists | 60 bytes |
| `AGENTS.md` | Exists | 3,351 bytes |
| `PROJECT_LOG.md` | Exists | 36,769 bytes |
| `report/REPORT_NOTES.md` | Exists | 25,024 bytes |
| `report/final_report_draft.md` | Exists | 22,481 bytes |
| `data/data_link.txt` | Exists | 462 bytes |

Scripts:

| Path | Status | Size |
| --- | --- | ---: |
| `src/preprocessing.py` | Exists | 3,004 bytes |
| `src/eda.py` | Exists | 14,728 bytes |
| `src/train.py` | Exists | 19,810 bytes |
| `src/evaluate.py` | Exists | 4,628 bytes |
| `src/stratified_baseline.py` | Exists | 8,613 bytes |
| `src/modeling.py` | Exists | 9,730 bytes |
| `demo/demo.py` | Exists | 6,171 bytes |

Processed/local generated files:

| Path | Status | Size |
| --- | --- | ---: |
| `data/processed/windowed_labeled_data.csv` | Exists | 175,958,810 bytes |
| `models/final_model.joblib` | Exists | 6,097 bytes |

Results:

| Path | Status | Size |
| --- | --- | ---: |
| `results/metrics_table.csv` | Exists | 3,896 bytes |
| `results/threshold_table.csv` | Exists | 1,741 bytes |
| `results/test_metrics.csv` | Exists | 404 bytes |
| `results/stratified_baseline_metrics.csv` | Exists | 2,380 bytes |
| `results/eda_summary.csv` | Exists | 1,119 bytes |
| `results/event_summary.csv` | Exists | 368 bytes |
| `results/demo_output.txt` | Exists | 926 bytes |
| `results/full_reproducibility_run.txt` | Exists | 15,514 bytes |

Plots:

| Path | Status | Size |
| --- | --- | ---: |
| `results/plots/class_balance.png` | Exists | 37,823 bytes |
| `results/plots/split_class_balance.png` | Exists | 54,628 bytes |
| `results/plots/sensor_correlation_heatmap.png` | Exists | 145,311 bytes |
| `results/plots/failure_windows_timeline.png` | Exists | 224,067 bytes |
| `results/plots/key_sensor_distributions.png` | Exists | 100,424 bytes |
| `results/plots/confusion_matrix.png` | Exists | 43,528 bytes |
| `results/plots/precision_recall_curve.png` | Exists | 61,866 bytes |
| `results/plots/validation_probability_distribution.png` | Exists | 56,335 bytes |
| `results/plots/test_probability_distribution.png` | Exists | 55,411 bytes |
| `results/plots/stratified_baseline_confusion_matrix.png` | Exists | 104,796 bytes |

Additional guideline-required item:

| Path | Status | Evidence |
| --- | --- | --- |
| `report/final_report.pdf` | Missing | `stat` failed with `No such file or directory` |

## Step 4 - Methodology Verification

| Requirement | Status | Evidence |
| --- | --- | --- |
| Supervised binary classification | Satisfied | `README.md`, `AGENTS.md`, `report/final_report_draft.md`, `src/train.py` |
| Target `0 = normal`, `1 = failure-risk/anomaly` | Satisfied | `README.md`, `AGENTS.md`, `src/config.py`, `src/labeling.py` |
| 1-hour early-warning label before documented failure start until failure end | Satisfied | `src/config.py`, `src/labeling.py`, `results/event_summary.csv` |
| 1-minute window feature engineering | Satisfied | `src/features.py`, `src/preprocessing.py`, full run log |
| Mean/std/min/max/last statistics per numeric sensor | Satisfied | `src/features.py` uses `.agg(["mean", "std", "min", "max", "last"])` |
| `window_start` excluded from model features | Satisfied | `src/modeling.py` drops `TARGET_COLUMN` and `WINDOW_START_COLUMN` in `split_features_labels` |
| `target` excluded from model features | Satisfied | `src/modeling.py`, `src/train.py`, full run log |
| Event-aware split is main realistic evaluation | Satisfied | `src/modeling.py`, `src/train.py`, `src/evaluate.py`, `README.md` |
| Train uses failure events 1-2 | Satisfied | `results/full_reproducibility_run.txt`, `results/event_summary.csv` |
| Validation uses failure event 3 | Satisfied | `results/full_reproducibility_run.txt`, `results/event_summary.csv` |
| Test uses failure event 4 | Satisfied | `results/full_reproducibility_run.txt`, `results/event_summary.csv` |
| Test set is not used for model selection | Satisfied | `src/train.py` selects on validation metrics before event-4 test evaluation |
| Threshold tuning uses validation only | Satisfied | `src/train.py`, full run log line "Threshold tuning uses validation probabilities only" |
| Stratified baseline is labeled optimistic only | Satisfied | `src/stratified_baseline.py`, `results/stratified_baseline_metrics.csv`, full run log |
| Model is not described as deployment-ready | Satisfied | `README.md`, `demo/demo.py`, `report/final_report_draft.md`, full run demo note |

## Step 5 - Model Requirements And Results

Exact five models used:

1. Logistic Regression baseline
2. Decision Tree
3. Random Forest
4. Extra Trees
5. HistGradientBoostingClassifier

Baseline model: Logistic Regression baseline.

Models with hyperparameter tuning:

- Decision Tree
- Random Forest
- HistGradientBoostingClassifier

Extra Trees and Logistic Regression were fit as fixed configured models in the main
event-aware workflow.

Best hyperparameters found:

| Model | Best hyperparameters |
| --- | --- |
| Logistic Regression baseline | Not grid-tuned |
| Decision Tree | `classifier__max_depth=4`, `classifier__min_samples_leaf=1` |
| Random Forest | `classifier__max_depth=8`, `classifier__min_samples_leaf=1` |
| Extra Trees | Not grid-tuned |
| HistGradientBoostingClassifier | `classifier__learning_rate=0.05`, `classifier__max_leaf_nodes=31` |

Validation metrics for each model at the validation-selected best-F1 threshold:

| Model | Threshold | Precision | Recall | F1 | F2 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression baseline | 0.61 | 0.723321 | 0.950736 | 0.821582 | 0.894490 | 0.957740 |
| Decision Tree | 0.13 | 0.989879 | 0.334588 | 0.500128 | 0.385647 | 0.345479 |
| Random Forest | 0.31 | 0.909091 | 0.396853 | 0.552513 | 0.447255 | 0.567006 |
| Extra Trees | 0.04 | 0.617140 | 0.795758 | 0.695158 | 0.752215 | 0.920710 |
| HistGradientBoostingClassifier | 0.11 | 0.970472 | 0.505987 | 0.665168 | 0.559549 | 0.848159 |

Best validation F2 thresholds:

| Model | Best F2 threshold | Best validation F2 |
| --- | ---: | ---: |
| Logistic Regression baseline | 0.61 | 0.894490 |
| Decision Tree | 0.13 | 0.385647 |
| Random Forest | 0.19 | 0.479038 |
| Extra Trees | 0.03 | 0.771677 |
| HistGradientBoostingClassifier | 0.11 | 0.559549 |

Final selected model:

| Item | Value |
| --- | --- |
| Selected model | Logistic Regression baseline |
| Selection strategy | `validation_f1_threshold_tuned` |
| Selected threshold | 0.61 |
| Evidence | `results/threshold_table.csv`, `results/test_metrics.csv`, `results/full_reproducibility_run.txt` |

Final held-out event-4 test metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9942902805882429 |
| Precision | 0.0 |
| Recall | 0.0 |
| F1 | 0.0 |
| F2 | 0.0 |
| ROC-AUC | 0.12087550368094527 |
| True negatives | 77,144 |
| False positives | 113 |
| False negatives | 330 |
| True positives | 0 |
| Predicted normal | 77,474 |
| Predicted failure-risk | 113 |

Interpretation: the selected event-aware model missed every true failure-risk window
in held-out event 4. Accuracy is high only because the event-4 test block is dominated
by normal windows.

Stratified baseline metrics:

| Model | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression baseline | 0.963616 | 0.357270 | 0.961538 | 0.520969 | 0.718494 | 0.984914 |
| Decision Tree | 0.995113 | 0.842105 | 0.938462 | 0.887676 | 0.917466 | 0.968694 |
| Random Forest | 0.996340 | 0.863830 | 0.975962 | 0.916479 | 0.951265 | 0.994999 |
| Extra Trees | 0.996340 | 0.869490 | 0.967308 | 0.915794 | 0.946022 | 0.992644 |
| HistGradientBoostingClassifier | 0.995489 | 0.830081 | 0.981731 | 0.899559 | 0.947124 | 0.998631 |

Why the stratified baseline is not the main deployment result:

- It uses a stratified random window-level split.
- It can mix 1-minute windows from the same failure events across train and test.
- The processed dataset has 5,200 positive windows but only 4 independent failure
  events.
- It measures feature separability under an optimistic split, not generalization to a
  future unseen failure event.
- The event-aware split remains the main realistic evaluation because it holds out
  failure event 4 entirely.

## Step 6 - COEN 330 Guideline Compliance

| Guideline item | Status | Evidence |
| --- | --- | --- |
| Problem definition | Complete | `README.md`, `report/final_report_draft.md`, `report/REPORT_NOTES.md` |
| Dataset description | Complete | `README.md`, `data/data_link.txt`, `report/final_report_draft.md`, `results/eda_summary.csv` |
| Preprocessing | Complete | `src/preprocessing.py`, `src/features.py`, `src/labeling.py`, full run log |
| EDA | Complete | `src/eda.py`, `results/eda_summary.csv`, `results/event_summary.csv`, `results/plots/` |
| Feature engineering | Complete | `src/features.py`, `src/preprocessing.py`, `report/final_report_draft.md` |
| Five models | Complete | `src/train.py`, `results/metrics_table.csv` |
| Baseline model | Complete | Logistic Regression baseline in `src/train.py` and result tables |
| Hyperparameter tuning | Complete | Decision Tree, Random Forest, and HistGradientBoostingClassifier tuned in `src/train.py`; this meets the explicit "at least three" requirement |
| Validation strategy | Complete | Event-aware blocked split in `src/modeling.py`; train events 1-2, validation event 3, test event 4 |
| Test separation | Complete | `src/train.py` selects model/threshold on validation; event-4 test evaluated after selection |
| Appropriate metrics | Partial | Recall, F2, F1, precision, accuracy, ROC-AUC, and confusion matrices exist; PR-AUC value is not exported, though a validation precision-recall curve plot exists |
| Model comparison table | Complete | `results/metrics_table.csv`, `results/threshold_table.csv`, `report/final_report_draft.md` |
| Error analysis | Partial | False negatives, false positives, event-4 failure, and limitations are discussed; no separate per-window qualitative example table or final feature-importance artifact exists |
| Limitations | Complete | `report/final_report_draft.md`, `report/REPORT_NOTES.md`, `README.md` |
| Reproducibility | Complete | Full run passed; `README.md`, `requirements.txt`, scripts, outputs, and `results/full_reproducibility_run.txt` exist |
| Demo | Complete | `demo/demo.py` passed and states non-deployment-ready status |
| Final report structure | Partial | `report/final_report_draft.md` has the 15 required sections, but `report/final_report.pdf` is missing |
| Team contribution section | Partial | Present as placeholders in `report/final_report_draft.md`; real names/contributions still needed |
| References | Complete | `report/final_report_draft.md` cites MetroPT-3/UCI, Python libraries, course guideline, and tools |
| Academic integrity / external tools acknowledgment | Complete | `report/final_report_draft.md` acknowledges external libraries and AI-assisted tools |

## Step 7 - Remaining Work

Priority 1 - must fix before report/PDF or final submission:

- Generate `report/final_report.pdf` from the draft and verify it uses only actual
  results.
- Replace all team contribution placeholders with final names and verified individual
  contributions.
- Keep the event-aware event-4 failure explicit in the final PDF: recall, F1, and F2
  are 0.0, with 330 false negatives and 0 true positives.
- Prepare the final Moodle ZIP only after the PDF is ready, excluding raw data,
  processed CSV files, virtual environments, and large model files.
- Decide whether to keep the rerun-modified tracked result CSVs. The only observed
  differences are last-decimal Random Forest probability values in `metrics_table.csv`
  and `threshold_table.csv`.

Priority 2 - should fix if time allows:

- Pin known-good dependency versions or add an environment note based on the observed
  successful environment.
- Add PR-AUC only if intentionally regenerating metrics without changing labels,
  failure windows, split logic, model selection, thresholds, or the current reported
  metrics.
- Add quantified outlier counts or per-window qualitative examples only if generated
  from existing data without changing the methodology.
- After generating the PDF, do a final guideline pass against
  `docs/COEN330_Project_Guidelines.pdf`.

Priority 3 - optional polish:

- Add a demo screenshot or short video if useful for submission.
- Prepare presentation slides only if requested by the instructor.
- Add a brief note in the README that the full reproducibility log is available in
  `results/full_reproducibility_run.txt`.

## Step 8 - Log Updates

Updated:

- `PROJECT_LOG.md`
- `report/REPORT_NOTES.md`

The appended summaries record:

- The review command sequence.
- That the full run passed.
- The main outputs confirmed.
- The final event-aware event-4 failure result.
- Remaining submission issues.

## Final Review Conclusion

The project is reproducible in the current local environment and has the required core
scripts, generated result files, plots, model artifact, processed local data, and demo.
The realistic event-aware evaluation remains the main finding and is negative for
held-out event 4. The largest remaining submission blockers are not code execution
problems; they are report/package completion problems: missing final PDF, placeholder
team contributions, and final ZIP preparation.
