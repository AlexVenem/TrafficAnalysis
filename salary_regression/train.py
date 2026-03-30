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


def train_model(x_path: str | Path, y_path: str | Path) -> dict[str, float | int]:
    x = load_features(x_path)
    y = load_target(y_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = build_model(random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    metrics: dict[str, float | int] = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "n_features": int(x.shape[1]),
    }

    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train salary regression model.")
    parser.add_argument("x_path", type=Path, help="Path to x_data.npy")
    parser.add_argument("y_path", type=Path, help="Path to y_data.npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(args.x_path, args.y_path)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()