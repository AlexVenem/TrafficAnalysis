from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def build_model(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )