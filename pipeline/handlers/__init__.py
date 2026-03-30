from __future__ import annotations
 
try:
    from .clean import CleanHandler
    from .encode import EncodeHandler
    from .export import ExportHandler
    from .impute import ImputeHandler
    from .load import LoadHandler
    from .parse import ParseHandler
    from .rename import RenameHandler
    from .scale import ScaleHandler
    from .target import TargetHandler
except ImportError:
    from pipeline.handlers.clean import CleanHandler        # type: ignore[no-redef]
    from pipeline.handlers.encode import EncodeHandler      # type: ignore[no-redef]
    from pipeline.handlers.export import ExportHandler      # type: ignore[no-redef]
    from pipeline.handlers.impute import ImputeHandler      # type: ignore[no-redef]
    from pipeline.handlers.load import LoadHandler          # type: ignore[no-redef]
    from pipeline.handlers.parse import ParseHandler        # type: ignore[no-redef]
    from pipeline.handlers.rename import RenameHandler      # type: ignore[no-redef]
    from pipeline.handlers.scale import ScaleHandler        # type: ignore[no-redef]
    from pipeline.handlers.target import TargetHandler      # type: ignore[no-redef]
 
__all__ = [
    "LoadHandler",    # 1. Чтение CSV
    "RenameHandler",  # 2. Нормализация заголовков
    "CleanHandler",   # 3. Очистка
    "ParseHandler",   # 4. Парсинг текстовых полей
    "TargetHandler",  # 5. Формирование целевой переменной
    "EncodeHandler",  # 6. OHE
    "ImputeHandler",  # 7. Заполнение пропусков
    "ScaleHandler",   # 8. Нормализация
    "ExportHandler",  # 9. Сохранение .npy
]