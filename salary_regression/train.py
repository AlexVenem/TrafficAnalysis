from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import METRICS_PATH, MODEL_PATH, RESOURCES_DIR
from .io_utils import load_features, load_target
from .model import build_model


def clip_target(y: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    lower = np.quantile(y, lower_q)
    upper = np.quantile(y, upper_q)
    return np.clip(y, lower, upper)


def train_model(x_path: str | Path, y_path: str | Path) -> dict[str, float | int]:
    x = load_features(x_path)
    y = load_target(y_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    y_train_clipped = clip_target(y_train, lower_q=0.01, upper_q=0.99)

    model = build_model()

    y_train_log = np.log1p(y_train_clipped)
    model.fit(x_train, y_train_log)

    pred_log = model.predict(x_test)
    predictions = np.expm1(pred_log)
    predictions = np.maximum(predictions, 0.0)

    metrics: dict[str, float | int] = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "n_features": int(x.shape[1]),
    }

    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ridge salary regression model.")
    parser.add_argument("x_path", type=Path, help="Path to x_data.npy")
    parser.add_argument("y_path", type=Path, help="Path to y_data.npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(args.x_path, args.y_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()