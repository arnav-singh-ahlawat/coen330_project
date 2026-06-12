# Demo

This folder contains a small command-line demo for the MetroPT-3 binary classifier.

Before running the demo:

1. Install dependencies with `pip install -r requirements.txt`.
2. Place the raw MetroPT-3 CSV file in `data/raw/`.
3. Configure or verify binary labels.
4. Train the model with `python -m src.train`.

Run:

```bash
python demo/demo.py
```

The demo prints predictions for a few local records. It does not include or commit raw dataset rows.
