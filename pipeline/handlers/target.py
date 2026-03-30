from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.base import Handler, PipelineContext

logger = logging.getLogger(__name__)


class TargetHandler(Handler):
    """
    Создаёт целевую колонку salary_class:

    - Удаляет строки без salary_rub.
    - Разбивает оставшиеся на 3 класса по квантилям (33% / 67%).
    - Устанавливает ctx.target_col = 'salary_class'.

    Tercile-разбивка выбрана, потому что зарплаты сильно скошены;
    это даёт сбалансированные классы без ручного порога.
    """

    def process(self, ctx: PipelineContext) -> PipelineContext:
        df = ctx.df

        if "salary_rub" not in df.columns:
            raise ValueError("Колонка salary_rub не найдена. ParseHandler выполнен?")

        # Удаляем строки без зарплаты
        before = len(df)
        df = df.dropna(subset=["salary_rub"]).copy()
        logger.info("Удалено строк без зарплаты: %d", before - len(df))

        # Границы tercile
        q33 = df["salary_rub"].quantile(0.33)
        q67 = df["salary_rub"].quantile(0.67)
        logger.info(
            "Границы классов: низкая < %.0f ≤ средняя < %.0f ≤ высокая",
            q33, q67,
        )

        def _classify(x: float) -> int:
            if x <= q33:
                return 0
            if x <= q67:
                return 1
            return 2

        df["salary_class"] = df["salary_rub"].apply(_classify).astype(np.int32)
        ctx.target_col = "salary_class"

        dist = df["salary_class"].value_counts().sort_index()
        logger.info("Распределение классов:\n%s", dist.to_string())

        ctx.df = df.reset_index(drop=True)
        return ctx
