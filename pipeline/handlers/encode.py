from __future__ import annotations

import logging

import pandas as pd

from pipeline.base import Handler, PipelineContext

logger = logging.getLogger(__name__)

# Максимальная кардинальность для OHE
_MAX_CARDINALITY = 50

# Колонки для OHE
_OHE_COLUMNS = ("city",)


class EncodeHandler(Handler):
    """
    Применяет OHE к колонке city (и любым другим строковым колонкам).

    Если кардинальность превышает _MAX_CARDINALITY — колонка удаляется
    с предупреждением.

    После шага все колонки в DataFrame числовые.
    Заполняет ctx.feature_cols всеми числовыми колонками кроме целевой.
    """

    def process(self, ctx: PipelineContext) -> PipelineContext:
        df = ctx.df

        for col in _OHE_COLUMNS:
            if col not in df.columns:
                continue

            n_unique = df[col].nunique(dropna=True)
            if n_unique > _MAX_CARDINALITY:
                logger.warning(
                    "'%s' имеет %d уникальных значений — колонка удалена (OHE > %d)",
                    col, n_unique, _MAX_CARDINALITY,
                )
                df.drop(columns=[col], inplace=True)
                continue

            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False, dtype=float)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            logger.info("OHE '%s' → %d бинарных признаков", col, len(dummies.columns))

        # Собираем feature_cols: всё числовое, кроме целевой и salary_rub
        exclude = {ctx.target_col, "salary_rub"}
        ctx.feature_cols = [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        ctx.df = df
        logger.info(
            "Признаков: %d → %s",
            len(ctx.feature_cols),
            ctx.feature_cols,
        )
        return ctx
