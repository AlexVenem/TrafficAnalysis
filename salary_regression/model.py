from __future__ import annotations

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]),
            ),
        ]
    )