from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCES_DIR = PACKAGE_ROOT / "resources"
MODEL_PATH = RESOURCES_DIR / "salary_regressor.pkl"
METRICS_PATH = RESOURCES_DIR / "metrics.json"