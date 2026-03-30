""" Заполнение пропущенных значений."""

from __future__ import annotations

import logging

import pandas as pd

from pipeline.base import Handler, PipelineContext

logger = logging.getLogger(__name__)


class ImputeHandler(Handler):
    """
    Заполняет пропуски в матрице признаков:

    - Бинарные OHE-колонки (все значения 0 или 1) → 0.
    - Остальные числовые → медиана колонки.
    """

    def process(self, ctx: PipelineContext) -> PipelineContext:
        df = ctx.df
        cols = ctx.feature_cols

        if not cols:
            logger.warning("feature_cols пусто, ImputeHandler пропущен")
            return ctx

        ohe = [c for c in cols if set(df[c].dropna().unique()).issubset({0.0, 1.0})]
        num = [c for c in cols if c not in ohe]

        if ohe:
            df[ohe] = df[ohe].fillna(0.0)

        for col in num:
            n_miss = int(df[col].isna().sum())
            if n_miss == 0:
                continue
            med = df[col].median()
            df[col] = df[col].fillna(med)
            logger.info("'%s': %d пропусков → медиана %.2f", col, n_miss, med)

        ctx.df = df
        return ctx
