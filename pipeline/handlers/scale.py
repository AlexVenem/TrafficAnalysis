from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import StandardScaler

from pipeline.base import Handler, PipelineContext

logger = logging.getLogger(__name__)


class ScaleHandler(Handler):
    """
    Нормализует числовые (не-бинарные) признаки: z = (x − μ) / σ.
    Бинарные OHE-признаки не масштабируются.

    Собирает итоговые массивы:
        ctx.x_data : float32 (N, F)
        ctx.y_data : fgloat
    """

    def process(self, ctx: PipelineContext) -> PipelineContext:
        df           = ctx.df
        feature_cols = ctx.feature_cols
        target_col   = ctx.target_col

        if not feature_cols:
            raise ValueError("feature_cols пусто — EncodeHandler выполнен?")
        if target_col not in df.columns:
            raise ValueError(f"Целевая колонка '{target_col}' не найдена.")

        # Делим признаки на бинарные и числовые
        ohe = [c for c in feature_cols
               if set(df[c].dropna().unique()).issubset({0.0, 1.0})]
        num = [c for c in feature_cols if c not in ohe]

        X_parts: list[np.ndarray] = []

        # Масштабируем числовые
        if num:
            X_num = df[num].values.astype(np.float32)
            scaler = StandardScaler()
            X_num = scaler.fit_transform(X_num).astype(np.float32)
            X_parts.append(X_num)
            ctx.meta["scaler_means"]  = scaler.mean_.tolist()
            ctx.meta["scaler_scales"] = scaler.scale_.tolist()
            ctx.meta["scaled_cols"]   = num
            logger.info("StandardScaler: %d колонок", len(num))

        # Бинарные — без изменений
        if ohe:
            X_parts.append(df[ohe].values.astype(np.float32))

        ctx.x_data = np.concatenate(X_parts, axis=1).astype(np.float32)
        ctx.y_data = df["salary_rub"].values.astype(np.float32)

        logger.info(
            "x_data %s float32  |  y_data %s int32",
            ctx.x_data.shape,
            ctx.y_data.shape,
        )
        return ctx
