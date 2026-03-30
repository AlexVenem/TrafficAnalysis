from __future__ import annotations

import logging

import pandas as pd

from pipeline.base import Handler, PipelineContext

logger = logging.getLogger(__name__)

# Колонки с долей пропусков выше порога будут удалены
_MAX_MISSING_RATIO = 0.95


class CleanHandler(Handler):
    """
    Базовая очистка DataFrame:

    1. Удаляет полностью пустые строки.
    2. Удаляет полные дубликаты.
    3. Удаляет колонки, где >95% значений — пропуски.
    4. Нормализует строки: strip, lower, пустую строку → NA.
    """

    def process(self, ctx: PipelineContext) -> PipelineContext:
        df = ctx.df
        n0 = len(df)

        # 1. Полностью пустые строки
        df.dropna(how="all", inplace=True)

        # 2. Дубликаты
        before = len(df)
        df.drop_duplicates(inplace=True)
        n_dup = before - len(df)
        if n_dup:
            logger.info("Удалено дубликатов: %d", n_dup)

        # 3. Почти пустые колонки
        miss = df.isnull().mean()
        drop_cols = miss[miss > _MAX_MISSING_RATIO].index.tolist()
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            logger.warning("Удалены колонки (>95%% пропусков): %s", drop_cols)

        # 4. Нормализация строк — итерируем по каждой колонке явно
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "string":
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .replace({"nan": pd.NA, "none": pd.NA, "не указано": pd.NA, "": pd.NA})
                )

        ctx.df = df.reset_index(drop=True)
        logger.info("Строк: %d → %d", n0, len(ctx.df))
        return ctx
