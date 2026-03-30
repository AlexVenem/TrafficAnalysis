from __future__ import annotations

from pathlib import Path

import joblib

from .config import MODEL_PATH
from .io_utils import load_features


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model weights were not found. "
            "Run training first: "
            "python -m salary_regression.train path/to/x_data.npy path/to/y_data.npy"
        )

    return joblib.load(MODEL_PATH)


def predict_salaries(x_path: str | Path) -> list[float]:
    x = load_features(x_path)
    model = load_model()
    predictions = model.predict(x)
    return [float(value) for value in predictions.tolist()]