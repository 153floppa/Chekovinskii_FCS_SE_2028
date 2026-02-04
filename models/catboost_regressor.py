"""
Модуль обучения и использования модели CatBoostRegressor для прогнозирования стоимости недвижимости.

Модель использует градиентный бустинг для регрессии и автоматически обрабатывает категориальные признаки.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
from utils import get_columns_grouped_by_dtypes, x_y_split
from config import RANDOM_SEED


def train_model(df: pd.DataFrame, target_column: str, val_size: float = 0.3, test_size: float = 0.2) -> CatBoostRegressor:
    """
    Обучает модель CatBoostRegressor для прогнозирования стоимости недвижимости.
    
    Args:
        df: DataFrame с данными для обучения
        target_column: Название целевой переменной (цена за кв.м)
        val_size: Доля данных для валидации (по умолчанию 0.3)
        test_size: Доля данных для тестирования (по умолчанию 0.2)
        
    Returns:
        Обученная модель CatBoostRegressor
        
    Raises:
        AssertionError: Если сумма val_size и test_size превышает 0.7
    """
    assert val_size + test_size <= 0.7, "Amount of data for train is really low"

    tmp_df, test_df = train_test_split(df, test_size=test_size, random_state=RANDOM_SEED)
    train_df, val_df = train_test_split(tmp_df, test_size=val_size * (1 - test_size), random_state=RANDOM_SEED)

    X_train, y_train = x_y_split(train_df, target_column)
    X_val, y_val = x_y_split(val_df, target_column)
    X_test, y_test = x_y_split(test_df, target_column)

    cat_cols = list(get_columns_grouped_by_dtypes(df)['object'])

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)

    model = CatBoostRegressor(
        iterations=10000,
        learning_rate=0.1,
        depth=8,
        random_seed=RANDOM_SEED,
        loss_function='RMSE',
        custom_metric='R2',
        eval_metric='R2',
        verbose=100
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=10,
    )

    model.save_model('catboost_regressor')
    test_score = model.score(test_pool)
    print(f"R² на тестовой выборке: {test_score:.4f}")
    
    return model


def get_model(model_path: str = 'catboost_regressor') -> CatBoostRegressor:
    """
    Загружает сохраненную модель CatBoostRegressor.
    
    Args:
        model_path: Путь к файлу с сохраненной моделью
        
    Returns:
        Загруженная модель CatBoostRegressor
    """
    model = CatBoostRegressor()
    return model.load_model(fname=model_path)
