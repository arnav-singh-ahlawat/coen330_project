# COEN 330 Project Guideline Checklist

Official source of truth: `docs/COEN330_Project_Guidelines.pdf`.

Use this checklist to verify the COEN 330 project package before submission. The current project is a supervised binary classification workflow for MetroPT-3 compressor failure-risk prediction.

## Problem Definition

- [ ] State the project title and applied domain.
- [ ] Formulate the problem clearly as a machine learning task.
- [ ] State the task type: supervised binary classification.
- [ ] Define the input features.
- [ ] Define the output/target variable.
- [ ] Define the target classes: `0 = normal operation`, `1 = failure-risk/anomaly`.
- [ ] Explain the motivation and application context.
- [ ] Explain the expected practical value of the solution.
- [ ] State assumptions and scope limitations.
- [ ] Avoid claiming deployment readiness.

## Dataset And Data Quality

- [ ] Identify the dataset source or collection method.
- [ ] Include dataset access and preparation instructions.
- [ ] Report the number of raw samples and features.
- [ ] Report the number of processed samples and features.
- [ ] Describe sensor/data types and time-series structure.
- [ ] Describe the target labels and how they are created.
- [ ] State the failure-risk label definition: one hour before documented failure start through documented failure end.
- [ ] Document class balance or target distribution.
- [ ] Document missing values and how they are handled.
- [ ] Document outliers or explain why outlier handling was limited.
- [ ] Document scaling, normalization, encoding, or time-series transformations.
- [ ] Document train/validation/test splitting.
- [ ] Discuss leakage risks and how they were avoided.
- [ ] Discuss ethical, privacy, licensing, or usage constraints if applicable.
- [ ] Keep raw and processed data out of Git commits.

## Exploratory Data Analysis

- [ ] Include feature distributions for important sensors or engineered features.
- [ ] Include class balance or target distribution.
- [ ] Include train/validation/test class balance.
- [ ] Include correlations or feature relationship plots.
- [ ] Include missing-value analysis.
- [ ] Include outlier analysis or a justified limitation.
- [ ] Include representative examples or visualizations when applicable.
- [ ] Include observations, not only figures.
- [ ] Explain expected modeling difficulty from EDA evidence.

## Feature Engineering And Data Representation

- [ ] Explain the one-minute window representation.
- [ ] Describe generated statistical features such as mean, standard deviation, minimum, maximum, and last value.
- [ ] Include any row-count or time-derived features if used.
- [ ] Explain which columns are excluded from model features.
- [ ] Confirm `window_start` is used only for ordering/splitting, not as a model feature.
- [ ] Confirm `target` is not used as a model feature.
- [ ] Justify why features were kept, removed, transformed, or created.
- [ ] Document imputation and scaling inside model pipelines.

## Model Development

- [ ] Implement and compare at least five suitable machine learning models.
- [ ] Include one simple, interpretable baseline model.
- [ ] Use Logistic Regression as the baseline.
- [ ] Include Decision Tree.
- [ ] Include Random Forest.
- [ ] Include Extra Trees.
- [ ] Include HistGradientBoostingClassifier.
- [ ] Justify why these models are appropriate for the task.
- [ ] Discuss model complexity, interpretability, and computational cost where relevant.

## Validation, Model Selection, And Hyperparameter Tuning

- [ ] Keep validation and test evaluation separated.
- [ ] Do not select models or thresholds using the test set.
- [ ] Use the event-aware blocked split as the main realistic evaluation.
- [ ] Use train events 1-2, validation event 3, and test event 4.
- [ ] Keep failure windows unchanged unless explicitly instructed.
- [ ] Keep event-aware split logic unchanged unless explicitly instructed.
- [ ] Tune at least three implemented models.
- [ ] Tune Decision Tree.
- [ ] Tune Random Forest.
- [ ] Tune HistGradientBoostingClassifier.
- [ ] Record tuned hyperparameters and selected values.
- [ ] Tune decision thresholds on validation data only.
- [ ] Treat the stratified window-level split as an optimistic baseline only.

## Evaluation

- [ ] Use metrics appropriate for imbalanced classification.
- [ ] Report accuracy, but do not treat it as the main metric.
- [ ] Emphasize recall for the failure-risk class.
- [ ] Emphasize F2-score.
- [ ] Report F1-score.
- [ ] Report precision.
- [ ] Report confusion matrices.
- [ ] Report ROC-AUC or PR-AUC where applicable.
- [ ] Include a clear comparison table of model results.
- [ ] Include final event-aware held-out event-4 test results.
- [ ] State honestly that the event-aware test failed to generalize to held-out failure event 4.
- [ ] Present stratified window-level results only as an optimistic comparison.

## Error Analysis And Interpretation

- [ ] Discuss false negatives and why they matter for predictive maintenance.
- [ ] Discuss false positives and their operational cost.
- [ ] Include examples of correct and incorrect predictions when applicable.
- [ ] Interpret important features or explain why feature interpretation is limited.
- [ ] Compare model behavior across candidate models.
- [ ] Explain why some models performed better or worse.
- [ ] Discuss limitations of the dataset and models.
- [ ] Discuss possible sources of error.
- [ ] Include future improvements.

## Reproducibility

- [ ] Provide a clear folder structure.
- [ ] Include `README.md` with setup and run instructions.
- [ ] Include `requirements.txt`, `environment.yml`, or equivalent dependencies.
- [ ] Include preprocessing code.
- [ ] Include training and validation code.
- [ ] Include evaluation code.
- [ ] Include EDA or visualization code.
- [ ] Include dataset link and preparation instructions.
- [ ] Include saved results and figures where appropriate.
- [ ] Use a clear random seed or explain randomness.
- [ ] Verify code runs from clean instructions before claiming completion.
- [ ] Do not commit raw data, processed data, virtual environments, or large model files.

## Demo Or Deployment Component

- [ ] Include a minimal usage demonstration.
- [ ] The demo may be a command-line script, notebook, local app, API, or simple web interface.
- [ ] Demo should load the trained model or reproduce a prediction workflow.
- [ ] Demo should use valid processed model features.
- [ ] Demo should show prediction output clearly.
- [ ] Demo should not take priority over sound validation and evaluation.
- [ ] Demo should state the current model is not deployment-ready.
- [ ] Consider adding a screenshot or short video if requested or useful.

## Final Report

- [ ] Write the final report in a clear technical style.
- [ ] Use IEEE-style formatting if practical.
- [ ] Target 8-10 pages excluding references and appendices.
- [ ] Submit the final report as a PDF.
- [ ] Include an Abstract.
- [ ] Include Introduction and motivation.
- [ ] Include Related work or background.
- [ ] Include Dataset description.
- [ ] Include Preprocessing and exploratory data analysis.
- [ ] Include Methodology and models.
- [ ] Include Validation and hyperparameter tuning strategy.
- [ ] Include Experimental setup.
- [ ] Include Results and model comparison.
- [ ] Include Error analysis and qualitative discussion.
- [ ] Include Demo or usage demonstration description.
- [ ] Include Limitations and future work.
- [ ] Include Conclusion.
- [ ] Include Team contribution section.
- [ ] Include References.
- [ ] Use actual verified results only.

## Submission Package

- [ ] Submit one compressed file on Moodle.
- [ ] Name the compressed file `TeamName_Project.zip`.
- [ ] Include the final report PDF.
- [ ] Include source code and notebooks/scripts for preprocessing, training, validation, evaluation, and visualization.
- [ ] Include the README.
- [ ] Include dataset information, link, description, access instructions, and preparation instructions.
- [ ] Include results and figures: tables, plots, confusion matrices, metrics, and qualitative examples.
- [ ] Include demo files, screenshots, or a short video link if large.
- [ ] Do not include raw data, processed CSV files, virtual environments, or large model files unless the instructor explicitly requires them.

## Team Contribution

- [ ] Add a team contribution section to the final report.
- [ ] List each member's concrete contributions.
- [ ] Cover data preparation, feature engineering, modeling, evaluation, repository management, demo, and report writing as applicable.

## References And Academic Integrity

- [ ] Cite the MetroPT-3 dataset source.
- [ ] Cite external code libraries, tutorials, papers, or documentation used.
- [ ] Acknowledge AI-assisted tools if used and required by instructor policy.
- [ ] Do not submit code, text, figures, or results that were not produced or properly acknowledged.
- [ ] Ensure team members can explain all submitted code and results.
- [ ] Keep a clear record of commands, results, and decisions in project logs.

## Grading Priorities From The PDF

- [ ] Problem definition and motivation: 10%.
- [ ] Dataset preparation, preprocessing, feature engineering, and EDA: 15%.
- [ ] Model development and validation, including hyperparameter tuning: 20%.
- [ ] Model evaluation and comparison: 20%.
- [ ] Error analysis and interpretation: 5%.
- [ ] Reproducibility and minimal usage demonstration: 10%.
- [ ] Final report quality: 10%.
- [ ] Presentation and Q&A if requested: 10%, or redistributed if no presentation is requested.

## Optional Bonus Items

- [ ] Original dataset collection.
- [ ] Advanced interpretability such as SHAP or LIME.
- [ ] Online deployment or polished user interface.
- [ ] Advanced model suitable for the task.
- [ ] Fairness, bias, or robustness analysis.
- [ ] Experiment tracking tools such as MLflow or Weights & Biases.
- [ ] Additional controlled experiments with feature sets, preprocessing strategies, model configurations, or validation settings.
