"""
Модуль утилит для предобработки и обработки данных.

Содержит функции для очистки данных, фильтрации и разделения на признаки и целевую переменную.
"""

import pandas as pd
from typing import Dict, Tuple, Optional, List
from config import SQUARE_COL


def delete_cols_with_any_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет колонки, содержащие хотя бы одно значение NaN.
    
    Args:
        df: Исходный DataFrame
        
    Returns:
        DataFrame без колонок с пропущенными значениями
    """
    return df.dropna(axis=1)


def delete_rows_with_big_square(df: pd.DataFrame, max_square: float = 100.0) -> pd.DataFrame:
    """
    Удаляет строки с площадью, превышающей заданное значение.
    
    Args:
        df: Исходный DataFrame
        max_square: Максимально допустимая площадь (по умолчанию 100 м²)
        
    Returns:
        DataFrame без строк с превышающей площадью
    """
    return df.drop(df[df[SQUARE_COL] > max_square].index)


def get_columns_grouped_by_dtypes(df: pd.DataFrame) -> Dict[str, pd.Index]:
    """
    Группирует колонки DataFrame по типам данных.
    
    Args:
        df: Исходный DataFrame
        
    Returns:
        Словарь с ключами 'int64', 'float64', 'object' и значениями - индексами колонок
        
    Raises:
        AssertionError: Если в DataFrame присутствуют типы данных, отличные от int64, float64, object
    """
    series_of_columns = df.columns.to_series()
    grouped_by_types = series_of_columns.groupby(df.dtypes).groups
    grouped_by_types = {k.name: v for k, v in grouped_by_types.items()}
    assert grouped_by_types.keys() == {'int64', 'float64', 'object'}, "Currently only 3 types is supported"
    return grouped_by_types


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

