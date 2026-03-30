from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from pipeline.base import Handler, PipelineContext

logger = logging.getLogger(__name__)



def _parse_gender_age(series: pd.Series) -> pd.DataFrame:
    """
    'Мужчина ,  42 года , родился 6 октября 1976'
    → gender (0=ж, 1=м), age (int)
    """
    gender = series.str.lower().str.contains("мужчина", na=False).astype(int)

    age = (
        series
        .str.extract(r",\s*(\d{1,3})\s*(?:год|лет)", expand=False)
        .apply(pd.to_numeric, errors="coerce")
    )
    return pd.DataFrame({"gender": gender, "age": age})


def _parse_salary(series: pd.Series) -> pd.Series:
    """
    '27 000 руб.' / '60 000 руб.' → float (рубли)
    Убирает пробелы внутри числа, затем конвертирует.
    """
    cleaned = (
        series
        .str.replace(r"\s+", "", regex=True)   # '27 000' → '27000'
        .str.extract(r"(\d+)", expand=False)
        .apply(pd.to_numeric, errors="coerce")
    )
    return cleaned


def _parse_city(series: pd.Series) -> pd.Series:
    """
    'Москва , м. Марьино , ...'  → 'москва'
    Берём первый токен до запятой, нижний регистр.
    """
    return series.str.split(",").str[0].str.strip().str.lower()


def _parse_employment(series: pd.Series) -> pd.DataFrame:
    """
    'частичная занятость, полная занятость'
    → emp_full (0/1), emp_part (0/1), emp_project (0/1)
    """
    s = series.str.lower().fillna("")
    return pd.DataFrame({
        "emp_full":    s.str.contains("полная").astype(int),
        "emp_part":    s.str.contains("частичная").astype(int),
        "emp_project": s.str.contains("проектная").astype(int),
    })


def _parse_schedule(series: pd.Series) -> pd.DataFrame:
    """
    'удаленная работа, гибкий график, полный день'
    → sched_fullday, sched_remote, sched_flex, sched_shift
    """
    s = series.str.lower().fillna("")
    return pd.DataFrame({
        "sched_fullday": s.str.contains("полный день").astype(int),
        "sched_remote":  s.str.contains("удален").astype(int),
        "sched_flex":    s.str.contains("гибкий").astype(int),
        "sched_shift":   s.str.contains("сменный").astype(int),
        "sched_vakhta":  s.str.contains("вахт").astype(int),
    })


def _parse_experience(series: pd.Series) -> pd.Series:
    """
    'Опыт работы 6 лет 1 месяц\\n\\n...'
    → общий опыт в месяцах (float)
    """
    years  = series.str.extract(r"(\d+)\s*лет",    expand=False).apply(pd.to_numeric, errors="coerce").fillna(0)
    months = series.str.extract(r"(\d+)\s*месяц",  expand=False).apply(pd.to_numeric, errors="coerce").fillna(0)
    return (years * 12 + months).replace(0, np.nan)


def _parse_education(series: pd.Series) -> pd.Series:
    """
    'Высшее образование 2003 МГУ...' / 'Среднее специальное...'
    → ordinal: 0=среднее, 1=среднее-специальное, 2=неполное высшее, 3=высшее
    """
    s = series.str.lower().fillna("")
    edu = pd.Series(0, index=series.index, dtype=float)
    edu = edu.where(~s.str.contains("среднее специальное|специальн"), 1)
    edu = edu.where(~s.str.contains("неполное высшее"),               2)
    edu = edu.where(~s.str.contains("высшее"),                        3)
    return edu


def _parse_has_car(series: pd.Series) -> pd.Series:
    """
    'Имеется собственный автомобиль' / 'Не указано' → 1 / 0
    """
    return series.str.lower().str.contains("автомобил", na=False).astype(int)


def _parse_updated_hour(series: pd.Series) -> pd.Series:
    """
    '28.04.2019 12:17' → час обновления (int 0-23)
    Косвенный признак активности кандидата.
    """
    return (
        pd.to_datetime(series, format="%d.%m.%Y %H:%M", errors="coerce")
        .dt.hour
        .astype(float)
    )


def _parse_title_len(series: pd.Series) -> pd.Series:
    """Длина строки желаемой должности — прокси для специализации."""
    return series.str.len().fillna(0).astype(float)


# Обработчик 

class ParseHandler(Handler):
    """
    Парсит все *_raw колонки в числовые/бинарные признаки.

    После этого шага DataFrame содержит только числа.
    Исходные *_raw колонки удаляются.
    """

    def process(self, ctx: PipelineContext) -> PipelineContext:
        df = ctx.df
        new_cols: dict[str, pd.Series | pd.DataFrame] = {}

        # gender_age_raw → gender, age
        if "gender_age_raw" in df.columns:
            new_cols["_ga"] = _parse_gender_age(df["gender_age_raw"])

        # salary_raw → salary_rub  (целевая переменная)
        if "salary_raw" in df.columns:
            new_cols["salary_rub"] = _parse_salary(df["salary_raw"])

        # title_raw → title_length
        if "title_raw" in df.columns:
            new_cols["title_length"] = _parse_title_len(df["title_raw"])

        # city_raw → city (строка, позже пойдёт в OHE)
        if "city_raw" in df.columns:
            df["city"] = _parse_city(df["city_raw"])

        # employment_raw → emp_*
        if "employment_raw" in df.columns:
            new_cols["_emp"] = _parse_employment(df["employment_raw"])

        # schedule_raw → sched_*
        if "schedule_raw" in df.columns:
            new_cols["_sched"] = _parse_schedule(df["schedule_raw"])

        # experience_raw → experience_months
        if "experience_raw" in df.columns:
            new_cols["experience_months"] = _parse_experience(df["experience_raw"])

        # education_raw → education_level
        if "education_raw" in df.columns:
            new_cols["education_level"] = _parse_education(df["education_raw"])

        # has_car_raw → has_car
        if "has_car_raw" in df.columns:
            new_cols["has_car"] = _parse_has_car(df["has_car_raw"])

        # updated_at_raw → update_hour
        if "updated_at_raw" in df.columns:
            new_cols["update_hour"] = _parse_updated_hour(df["updated_at_raw"])

        # Вставляем в DataFrame
        for key, val in new_cols.items():
            if isinstance(val, pd.DataFrame):
                for col in val.columns:
                    df[col] = val[col].values
            else:
                df[key] = val.values

        # Удаляем все *_raw колонки
        raw_cols = [c for c in df.columns if c.endswith("_raw")]
        df.drop(columns=raw_cols, inplace=True)

        ctx.df = df.reset_index(drop=True)
        logger.info("Признаки после парсинга: %s", list(ctx.df.columns))
        return ctx
