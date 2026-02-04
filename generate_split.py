"""
Модуль оптимизации дробления помещений для максимизации суммарной стоимости.

Реализует рекурсивный алгоритм поиска оптимального варианта разделения большого помещения
на несколько меньших с учетом ограничений и затрат на создание новых объектов.
"""

from catboost import Pool, CatBoostRegressor
from pandas import Series
from typing import Tuple, List
from copy import deepcopy
from config import (
    POSSIBLE_SQUARES,
    PRICE_PER_OBJECT,
    SQUARE_COL,
    MINIMAL_SQUARE,
    SPLIT_LIMIT,
    OWN_ENTRANCE_LIMIT,
)


def score(obj: Series, model: CatBoostRegressor, cat_cols: List[str]) -> float:
    """
    Предсказывает стоимость объекта недвижимости с помощью модели.
    
    Args:
        obj: Series с характеристиками объекта
        model: Обученная модель CatBoostRegressor
        cat_cols: Список названий категориальных признаков
        
    Returns:
        Предсказанная стоимость объекта (цена за кв.м × площадь)
    """
    obj_pool = Pool(obj.to_frame().T, cat_features=cat_cols)
    price_per_sqm = model.predict(obj_pool)[0]
    return price_per_sqm * obj[SQUARE_COL]


def maximize(
    obj: Series,
    limit_new: int,
    limit_own_entrance: int,
    model: CatBoostRegressor,
    cat_cols: List[str]
) -> Tuple[float, List[Series]]:
    """
    Рекурсивно находит оптимальный вариант дробления помещения.
    
    Алгоритм перебирает все возможные варианты разделения помещения на части,
    учитывая ограничения на количество новых объектов и отдельные входы.
    
    Args:
        obj: Series с характеристиками исходного объекта
        limit_new: Оставшееся количество разрешенных новых объектов
        limit_own_entrance: Оставшееся количество разрешенных отдельных входов
        model: Обученная модель для предсказания стоимости
        cat_cols: Список названий категориальных признаков
        
    Returns:
        Кортеж (максимальная суммарная стоимость, список новых объектов)
    """
    best_one = score(obj, model, cat_cols), [obj]

    if limit_new == 0:
        return best_one

    for new_square in POSSIBLE_SQUARES:
        init_square = obj[SQUARE_COL]
        if init_square - new_square < MINIMAL_SQUARE:
            continue

        new_obj = deepcopy(obj)
        new_obj[SQUARE_COL] = new_square
        new_limit_own_entrance = max(0, limit_own_entrance - 1)

        other_obj = deepcopy(obj)
        other_obj[SQUARE_COL] -= new_square

        new_res = maximize(other_obj, limit_new - 1, new_limit_own_entrance, model, cat_cols)
        new_price = new_res[0] + score(new_obj, model, cat_cols) - PRICE_PER_OBJECT
        new_objects = new_res[1] + [new_obj]

        if new_price > best_one[0]:
            best_one = new_price, new_objects

    return best_one


def split(obj: Series, model: CatBoostRegressor, cat_cols: List[str]) -> Tuple[float, List[Series]]:
    """
    Находит оптимальный вариант дробления помещения для максимизации стоимости.
    
    Функция является точкой входа для алгоритма оптимизации дробления.
    Учитывает ограничения из конфигурации (SPLIT_LIMIT, OWN_ENTRANCE_LIMIT).
    
    Args:
        obj: Series с характеристиками исходного объекта
        model: Обученная модель CatBoostRegressor
        cat_cols: Список названий категориальных признаков
        
    Returns:
        Кортеж (максимальная суммарная стоимость после дробления, список новых объектов)
    """
    return maximize(obj, SPLIT_LIMIT, OWN_ENTRANCE_LIMIT - 1, model, cat_cols)
