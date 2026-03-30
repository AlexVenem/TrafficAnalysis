"""
Сборщик цепочки.
"""

from __future__ import annotations

from .base import Handler
from .handlers import (
    CleanHandler,
    EncodeHandler,
    ExportHandler,
    ImputeHandler,
    LoadHandler,
    ParseHandler,
    RenameHandler,
    ScaleHandler,
    TargetHandler,
)

_STEPS: list[type[Handler]] = [
    LoadHandler,     # 1. Читаем CSV
    RenameHandler,   # 2. Нормализуем имена колонок
    CleanHandler,    # 3. Дубликаты, пустые строки, нормализация строк
    ParseHandler,    # 4. Парсим все текстовые поля в числа
    TargetHandler,   # 5. Формируем salary_class, удаляем без зарплаты
    EncodeHandler,   # 6. OHE для city, собираем feature_cols
    ImputeHandler,   # 7. Заполняем пропуски
    ScaleHandler,    # 8. StandardScaler, собираем x_data / y_data
    ExportHandler,   # 9. Сохраняем .npy
]


def build_chain() -> Handler:
    """
    Возвращает голову цепочки (первый обработчик).
    """
    instances = [cls() for cls in _STEPS]

    for current, nxt in zip(instances, instances[1:]):
        current.set_next(nxt)

    return instances[0]