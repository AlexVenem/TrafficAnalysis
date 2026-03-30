"""Загрузка CSV с автоопределением кодировки."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pipeline.base import Handler, PipelineContext

logger = logging.getLogger(__name__)

_ENCODINGS = ("utf-8", "utf-8-sig", "cp1251", "latin-1")


class LoadHandler(Handler):
    """
    Читает CSV в DataFrame, перебирая кодировки.
    Типично для выгрузок hh.ru встречается и UTF-8, и Windows-1251.
    """

    def process(self, ctx: PipelineContext) -> PipelineContext:
        path = ctx.csv_path
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        last_err: Exception | None = None
        for enc in _ENCODINGS:
            try:
                ctx.df = pd.read_csv(path, encoding=enc, low_memory=False)
                ctx.meta["encoding"] = enc
                logger.info("Прочитано %d строк, кодировка=%s", len(ctx.df), enc)
                return ctx
            except UnicodeDecodeError as exc:
                last_err = exc

        raise ValueError(
            f"Не удалось прочитать {path}. Последняя ошибка: {last_err}"
        )