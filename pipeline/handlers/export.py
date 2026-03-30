"""Сохранение x_data.npy и y_data.npy рядом с исходным CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pipeline.base import Handler, PipelineContext

logger = logging.getLogger(__name__)


class ExportHandler(Handler):
    """
    Cохраняет результаты на диск и верифицирует файлы.

    Создаёт:
        <output_dir>/x_data.npy  — матрица признаков (N, F), float32
        <output_dir>/y_data.npy  — вектор меток    (N,),    int32
    """

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.x_data is None or ctx.y_data is None:
            raise ValueError(
                "x_data / y_data не заполнены. ScaleHandler выполнен?"
            )

        out = Path(ctx.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        x_path = out / "x_data.npy"
        y_path = out / "y_data.npy"

        np.save(x_path, ctx.x_data)
        np.save(y_path, ctx.y_data)

        # Верификация: перечитываем и проверяем форму + dtype
        x_check = np.load(x_path)
        y_check = np.load(y_path)

        assert x_check.shape == ctx.x_data.shape, "Верификация x_data: форма не совпадает"
        assert y_check.shape == ctx.y_data.shape, "Верификация y_data: форма не совпадает"
        assert x_check.dtype == np.float32,        "Верификация x_data: dtype != float32"
        assert y_check.dtype == np.float32,           "Верификация y_data: dtype != float32"

        logger.info("✓ x_data.npy  %s  float32  → %s", ctx.x_data.shape, x_path)
        logger.info("✓ y_data.npy  %s  float32    → %s", ctx.y_data.shape, y_path)

        return ctx
