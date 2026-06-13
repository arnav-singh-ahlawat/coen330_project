# Demo

This folder contains a small command-line demo for the MetroPT-3 failure-risk classifier.
The demo loads one or two rows from the processed 1-minute window dataset and runs the
saved final model with the saved final threshold.

Before running the demo:

1. Install dependencies with `pip install -r requirements.txt`.
2. Generate `data/processed/windowed_labeled_data.csv` with `python -m src.preprocessing`.
3. Train the model with `python -m src.train` so `models/final_model.joblib` and
   `results/threshold_table.csv` exist.

Run:

```bash
python demo/demo.py
```

The demo prints the sample type, predicted probability of failure-risk, threshold,
predicted class, and true class for a normal window and a failure-risk window when both
are available. It does not create fake raw sensor features and does not include or
commit raw or processed dataset rows.

Note: This demo illustrates the pipeline usage on processed 1-minute windows. The
current model is not deployment-ready because event-aware testing showed poor
generalization to held-out failure event 4.
