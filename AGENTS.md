# COEN 330 Project Instructions For Codex

This repository is for a COEN 330 Applied Machine Learning course project.

Official guideline source of truth:

- `docs/COEN330_Project_Guidelines.pdf`
- Derived checklist: `docs/guideline_checklist.md`
- Current repository audit: `docs/guideline_audit.md`

Project title:

- Failure-Risk Prediction for a Metro Train Compressor Using Machine Learning

Main task:

- Supervised binary classification using the MetroPT-3 dataset.
- `0 = normal operation`
- `1 = failure-risk/anomaly`

Label definition:

- A 1-minute window is labeled failure-risk if its timestamp occurs from 1 hour before a documented failure start time until the documented failure end time.

Required methodology rules:

1. Do not invent results.
2. Do not tune on the test set.
3. Do not change failure windows unless explicitly instructed.
4. Do not change the event-aware split unless explicitly instructed.
5. Keep the event-aware evaluation as the main realistic evaluation.
6. Treat the stratified window-level split as an optimistic baseline only.
7. Do not claim the model is deployment-ready.
8. Always be honest that the event-aware test failed to generalize to held-out failure event 4.
9. Do not commit raw data, processed CSV files, virtual environments, or large model files.

Main evaluation:

- Event-aware blocked split.
- Train: failure events 1-2.
- Validation: failure event 3.
- Test: failure event 4.

Secondary evaluation:

- Stratified window-level baseline only.
- Describe it as an optimistic comparison, not deployment-level performance.

Required models:

1. Logistic Regression baseline.
2. Decision Tree.
3. Random Forest.
4. Extra Trees.
5. HistGradientBoostingClassifier.

Required tuning:

- Tune at least Decision Tree, Random Forest, and HistGradientBoostingClassifier.
- Use validation data only for model and threshold selection.

Metric priorities:

- Emphasize recall, F2-score, F1-score, precision, and confusion matrix.
- Report accuracy, but do not treat it as the main metric.
- Report ROC-AUC or PR-AUC where applicable.

Required project components:

- `README.md`
- `requirements.txt`
- preprocessing script
- training and validation script
- evaluation script
- EDA script and plots
- demo script
- dataset source/instructions
- results tables and plots
- project log
- final report draft/PDF
- team contribution section
- references and source/tool acknowledgments

Final report must include:

1. Abstract.
2. Introduction and motivation.
3. Related work or background.
4. Dataset description.
5. Preprocessing and exploratory data analysis.
6. Methodology and models.
7. Validation and hyperparameter tuning strategy.
8. Experimental setup.
9. Results and model comparison.
10. Error analysis and qualitative discussion.
11. Demo or usage demonstration.
12. Limitations and future work.
13. Conclusion.
14. Team contribution section.
15. References.

Every major change must update:

- `PROJECT_LOG.md`
- `report/REPORT_NOTES.md`
- `report/final_report_draft.md` when report content changes

Before claiming the project is complete, verify:

- code runs from clean instructions
- results files exist
- plots exist
- demo works
- README explains how to reproduce the project
- final report uses actual results only
- final report PDF exists
- team contribution and references are complete
