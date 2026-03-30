from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32 | np.float64]


def load_features(path: str | Path) -> FloatArray:
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file was not found: {feature_path}")

    data = np.load(feature_path)
    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix, got array with shape {data.shape}."
        )

    return data.astype(np.float64, copy=False)


def load_target(path: str | Path) -> FloatArray:
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Target file was not found: {target_path}")

    data = np.load(target_path)
    if data.ndim != 1:
        raise ValueError(
            f"Expected a 1D target vector, got array with shape {data.shape}."
        )

    return data.astype(np.float64, copy=False)