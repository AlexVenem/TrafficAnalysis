from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

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

    pred_log = model.predict(x)
    predictions = np.expm1(pred_log)
    predictions = np.maximum(predictions, 0.0)

    return [float(value) for value in predictions.tolist()]