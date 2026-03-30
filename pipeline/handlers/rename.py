from __future__ import annotations

import logging
import re

from pipeline.base import Handler, PipelineContext

logger = logging.getLogger(__name__)


_PATTERNS: list[tuple[str, str]] = [
    (r"опыт",                   "experience_raw"),   # «Опыт»
    (r"последен.*место",        "last_company_raw"), # «Последнее/нынешнее место работы»
    (r"последен.*должн",        "last_position_raw"),# «Последняя/нынешняя должность»
    (r"образован",              "education_raw"),
    (r"обновлен",               "updated_at_raw"),
    (r"занятость",              "employment_raw"),
    (r"график",                 "schedule_raw"),
    (r"ищет",                   "title_raw"),
    (r"город",                  "city_raw"),
    (r"авто",                   "has_car_raw"),
    (r"^зп$|зарплата|^зп\b",   "salary_raw"),        
    (r"пол.*возраст|возраст.*пол|^пол,", "gender_age_raw"), # «Пол, возраст»
    (r"unnamed",                "_row_index"),
]

_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), canonical)
    for pat, canonical in _PATTERNS
]


def _find_canonical(col: str) -> str | None:
    """Возвращает первое совпавшее каноническое имя или None."""
    col_lower = col.lower()
    for pattern, canonical in _COMPILED:
        if pattern.search(col_lower):
            return canonical
    return None


class RenameHandler(Handler):
    def process(self, ctx: PipelineContext) -> PipelineContext:
        ctx.meta["original_columns"] = list(ctx.df.columns)

        rename_map: dict[str, str] = {}
        used_names: dict[str, int] = {}
        unknown_counter = 0

        for col in ctx.df.columns:
            canonical = _find_canonical(col)

            if canonical:
                count = used_names.get(canonical, 0)
                used_names[canonical] = count + 1
                rename_map[col] = canonical if count == 0 else f"{canonical}_{count + 1}"
            else:
                safe = re.sub(r"[^\w]", "_", col.strip().lower())
                safe = re.sub(r"_+", "_", safe).strip("_") or f"col_{unknown_counter}"
                rename_map[col] = safe
                unknown_counter += 1

        ctx.df.rename(columns=rename_map, inplace=True)

        if "_row_index" in ctx.df.columns:
            ctx.df.drop(columns=["_row_index"], inplace=True)

        logger.info("Колонки после переименования: %s", list(ctx.df.columns))
        ctx.meta["rename_map"] = rename_map
        return ctx
