# COEN 330 MetroPT-3 Machine Learning Project

This repository contains a supervised binary classification workflow for the MetroPT-3 air compressor dataset.

Project task: classify each record as either:

- `0`: normal operation
- `1`: failure-risk/anomaly

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

Project decisions, reproducibility notes, commands, problems, fixes, and pending results are tracked in `PROJECT_LOG.md`. Report-ready explanations are collected in `report/REPORT_NOTES.md`.

Train a baseline model:

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

This is a supervised classification project, so training requires binary labels. The code supports:

1. An existing binary label column in the dataset.
2. Timestamp-based failure-risk windows configured in `src/config.py`.

Do not report final model performance until the labeling method, train/test split, and evaluation results are verified.
