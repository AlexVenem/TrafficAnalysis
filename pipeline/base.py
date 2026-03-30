from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """
    Изменяемый контейнер состояния пайплайна.

    Attributes:
        csv_path:     Путь к исходному CSV-файлу.
        df:           Рабочий DataFrame; обновляется каждым обработчиком.
        feature_cols: Имена колонок-признаков (заполняет EncodeHandler).
        target_col:   Имя целевой колонки.
        x_data:       Итоговая матрица признаков float32 (N, F).
        y_data:       Итоговый вектор меток int32 (N,).
        output_dir:   Куда сохранять .npy (по умолчанию — рядом с CSV).
        meta:         Произвольный словарь для обмена данными между шагами.
    """

    csv_path: Path
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_cols: list[str] = field(default_factory=list)
    target_col: str = "salary_rub"
    x_data: Optional[np.ndarray] = None
    y_data: Optional[np.ndarray] = None
    output_dir: Optional[Path] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.csv_path = Path(self.csv_path)
        if self.output_dir is None:
            self.output_dir = self.csv_path.parent


class Handler(ABC):
    """
    Абстрактный обработчик цепочки ответственности.

    Использование:
        handler_a.set_next(handler_b).set_next(handler_c)
        result_ctx = handler_a.handle(ctx)

    Каждый наследник реализует только метод `process`.
    Передача контекста дальше по цепочке — задача базового класса.
    """

    def __init__(self) -> None:
        self._next: Optional[Handler] = None


    def set_next(self, handler: Handler) -> Handler:
        """Присоединяет следующий обработчик; возвращает его."""
        self._next = handler
        return handler


    def handle(self, ctx: PipelineContext) -> PipelineContext:
        """Выполняет текущий шаг, логирует результат, передаёт дальше."""
        name = type(self).__name__
        logger.info("▶ %s", name)

        ctx = self.process(ctx)

        logger.info(
            "✓ %s  [строк=%d, колонок=%d]",
            name,
            len(ctx.df),
            len(ctx.df.columns),
        )

        if self._next is not None:
            return self._next.handle(ctx)
        return ctx


    @abstractmethod
    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Выполняет логику шага и возвращает обновлённый контекст."""
