"""
Модуль утилит для предобработки и обработки данных.

Содержит функции для очистки данных, фильтрации и разделения на признаки и целевую переменную.
"""

import pandas as pd
from typing import Dict, Tuple, Optional, List
from config import SQUARE_COL


def delete_cols_with_any_nan(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Очищает датафрейм: удаляет бесполезные колонки, заполняет NaN.

    Args:
        df: Исходный DataFrame
        threshold: Доля NaN для удаления колонки (default 0.5 = 50%)

    Returns:
        Очищенный DataFrame без бесполезных колонок и с заполненными NaN
    """
    df = df.copy()

    # Колонки которые точно не нужны
    useless_cols = ['Долгота', 'Широта', 'Ссылки на картинки', 'Прайс-лист']
    df = df.drop(columns=[c for c in useless_cols if c in df.columns], errors='ignore')

    # Удаляем только колонки с долей NaN больше threshold
    nan_ratio = df.isna().sum() / len(df)
    cols_to_keep = nan_ratio[nan_ratio <= threshold].index
    df = df[cols_to_keep]

    # Заполняем NaN в оставшихся колонках
    for col in df.columns:
        if df[col].isna().any():
            # Для текстовых колонок ставим "Неизвестно"
            if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                df[col] = df[col].fillna('Неизвестно')
            else:
                # Для числовых колонок ставим медиану (если возможно)
                try:
                    df[col] = df[col].fillna(df[col].median())
                except (TypeError, ValueError):
                    # Если медиану не получается считать, ставим 0
                    df[col] = df[col].fillna(0)

    return df


def get_columns_grouped_by_dtypes(df: pd.DataFrame) -> Dict[str, pd.Index]:
    """
    Группирует колонки DataFrame по типам данных.
    Нормализует string типы в object для совместимости.

    Args:
        df: Исходный DataFrame

    Returns:
        Словарь с ключами 'int64', 'float64', 'object' и значениями - индексами колонок
    """
    grouped_by_types = {'int64': [], 'float64': [], 'object': []}

    for col in df.columns:
        dtype_name = df[col].dtype.name

        # Нормализуем string и category в object
        if dtype_name in ('string', 'category'):
            grouped_by_types['object'].append(col)
        elif dtype_name == 'int64':
            grouped_by_types['int64'].append(col)
        elif dtype_name == 'float64':
            grouped_by_types['float64'].append(col)
        elif 'int' in dtype_name:  # int32, int16, etc.
            grouped_by_types['int64'].append(col)
        elif 'float' in dtype_name:  # float32, float16, etc.
            grouped_by_types['float64'].append(col)
        else:
            grouped_by_types['object'].append(col)

    # Конвертируем списки в pd.Index
    return {k: pd.Index(v) for k, v in grouped_by_types.items()}


def split_data(data: list, parts: int = 2) -> list:
    """
    Разбивает список на указанное количество приблизительно равных частей.

    Args:
        data: Исходный список объектов
        parts: Количество частей (1, 2, 3 или 4)

    Returns:
        Список с подсписками примерно равного размера

    Example:
        >>> data = list(range(10))
        >>> split_data(data, 2)
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        >>> split_data(data, 3)
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    """
    if parts < 1 or parts > 4:
        raise ValueError("parts должно быть от 1 до 4")

    n = len(data)
    if parts == 1:
        return [data]

    # Базовый размер каждой части
    base_size = n // parts
    remainder = n % parts

    # Первые remainder частей получают на 1 элемент больше
    result = []
    start = 0
    for i in range(parts):
        size = base_size + (1 if i < remainder else 0)
        result.append(data[start:start + size])
        start += size

    return result


def x_y_split(df: pd.DataFrame, target_column: str, use_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Разделяет DataFrame на матрицу признаков X и целевую переменную y.

    Args:
        df: Исходный DataFrame
        target_column: Название колонки с целевой переменной
        use_cols: Опциональный список колонок для использования (если None, используются все кроме target_column)

    Returns:
        Кортеж (X, y), где X - DataFrame с признаками, y - Series с целевой переменной
    """
    if use_cols:
        df = df[use_cols]
    X = df.loc[:, df.columns != target_column]
    y = df[target_column]
    return X, y

