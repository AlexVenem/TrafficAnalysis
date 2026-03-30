from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from salary_regression.predict import predict_salaries
else:
    from .predict import predict_salaries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Predict salaries from hh.ru pipeline output (.npy).",
    )
    parser.add_argument("x_path", type=Path, help="Path to x_data.npy")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    salaries = predict_salaries(args.x_path)
    print(json.dumps(salaries, ensure_ascii=False))


if __name__ == "__main__":
    main()
