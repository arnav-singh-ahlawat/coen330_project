"""Train a baseline supervised binary classifier for MetroPT-3."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    MODEL_FILE,
    PROCESSED_DATA_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loading import load_raw_data, standardize_columns
from src.features import build_features, save_processed_features
from src.labeling import add_target
from src.preprocessing import build_preprocessor
from src.utils import ensure_directories


def build_model():
    """Create the baseline classification pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train():
    """Train the model and save it to disk."""
    ensure_directories([MODEL_FILE.parent, PROCESSED_DATA_FILE.parent])

    raw = standardize_columns(load_raw_data())
    labeled = add_target(raw)
    X, y = build_features(labeled)
    save_processed_features(X, y, PROCESSED_DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions, target_names=["normal", "failure_risk"]))
    print(f"Saved model to {MODEL_FILE}")
    return model


if __name__ == "__main__":
    train()
