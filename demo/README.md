# Demo

This folder contains a small command-line demo for the MetroPT-3 failure-risk classifier.

Before running the demo:

1. Install dependencies with `pip install -r requirements.txt`.
2. Place the raw MetroPT-3 CSV file in `data/raw/`.
3. Configure or verify binary labels.
4. Train the model with `python -m src.train`.

Run:

```bash
python demo/demo.py
```

The demo prints normal versus failure-risk/anomaly predictions for a few local samples. It does not include or commit raw dataset rows.
