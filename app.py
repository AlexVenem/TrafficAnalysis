"""
Точка входа CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
 
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline import PipelineContext, build_chain


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app",
        description=(
            "Пайплайн обработки резюме hh.ru.\n"
            "Создаёт x_data.npy и y_data.npy рядом с исходным CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Путь к CSV-файлу с данными hh.ru",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Директория для .npy файлов (по умолчанию — папка CSV)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Подробные DEBUG-логи",
    )
    return parser.parse_args(argv)


def run(csv_path: Path, output_dir: Path | None = None) -> PipelineContext:
    """
    Запускает пайплайн и возвращает финальный контекст.
    Может использоваться как библиотечная функция (без CLI).
    """
    ctx = PipelineContext(csv_path=csv_path, output_dir=output_dir)
    chain = build_chain()
    return chain.handle(ctx)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("Запуск пайплайна: %s", args.csv_path)

    try:
        ctx = run(args.csv_path, output_dir=args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Ошибка: %s", exc)
        sys.exit(1)

    out = Path(ctx.output_dir)
    print(f"\n{'─'*50}")
    print(f"  x_data.npy  →  {out / 'x_data.npy'}  {ctx.x_data.shape}")
    print(f"  y_data.npy  →  {out / 'y_data.npy'}  {ctx.y_data.shape}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()