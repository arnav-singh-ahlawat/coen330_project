# Failure-Risk Prediction for a Metro Train Compressor Using Machine Learning

This repository contains a supervised binary classification workflow for the MetroPT-3 air compressor dataset. The project is framed as predictive maintenance for a metro train compressor.

The goal is to predict whether a 1-minute compressor sensor window belongs to a failure-risk/anomaly state. Compressor failures in railway systems can cause high repair cost, service disruption, and operational risk, so the model should prioritize catching failure-risk windows over maximizing overall accuracy.

Project task: classify each 1-minute window as either:

- `0`: normal operation
- `1`: failure-risk/anomaly

Failure-risk labels are defined from documented failure intervals. A 1-minute window is labeled `1` if its timestamp occurs from 1 hour before a documented failure start until the documented failure end. This creates a risk-state/early-warning label, not merely a current-failure label.

The raw MetroPT-3 dataset is intentionally not committed to GitHub. Keep raw files in `data/raw/`.

## Repository Structure

```text
data/
  raw/             # local-only raw dataset files
  processed/       # generated processed datasets
demo/              # small demo script
models/            # generated trained model artifacts
notebooks/         # exploratory notebooks
report/            # report outline and draft
results/plots/     # generated figures
src/               # project source code
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the MetroPT-3 dataset from the UCI Machine Learning Repository and place the raw CSV file in:

```text
data/raw/
```

See `data/data_link.txt` for the dataset source.

## Workflow

Project decisions, reproducibility notes, commands, problems, fixes, and verified results are tracked in `PROJECT_LOG.md`. Report-ready explanations are collected in `report/REPORT_NOTES.md`.

Run preprocessing:

```bash
python -m src.preprocessing
```

Train and compare the supervised models:

```bash
python -m src.train
```

Evaluate the saved model:

```bash
python -m src.evaluate
```

Run the demo:

```bash
python demo/demo.py
```

## Labeling Note

This is a supervised classification project, so training requires binary labels. The current pipeline uses timestamp-based failure-risk windows configured in `src/config.py`.

Do not change the configured failure-risk windows or event-aware held-out test split when reporting the current results.

## Evaluation Framing

Because false negatives are costly in predictive maintenance, the report should emphasize recall, F2-score, F1-score, and the confusion matrix for class `1`. Accuracy is still reported, but it is not the main success metric because the processed dataset is highly imbalanced.

The final evaluation uses an event-aware split: failure events 1 and 2 are used for training, event 3 for validation, and event 4 for testing. This is more realistic than a random row-level split because nearby windows from the same operating period or same failure event should not appear in both training and test data.

The verified held-out event-4 test result shows poor generalization: recall, F2-score, and F1-score are all `0.0`, with 330 false negatives and 0 true positives for the failure-risk class. This should be presented as a major dataset/model limitation caused by supervised learning with only four independent failure events, not as a coding error or deployment-ready result.
