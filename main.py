"""
Модуль обучения модели машинного обучения для прогнозирования стоимости коммерческой недвижимости.

Основные функции:
- Загрузка и предобработка данных
- Обучение модели CatBoostRegressor
- Анализ важности признаков
- Визуализация результатов
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from utils import delete_cols_with_any_nan, get_columns_grouped_by_dtypes
from models.catboost_regressor import train_model
from config import PRICE_COL


def load_data(filepath: str = 'data/result.pkl') -> pd.DataFrame:
    """
    Загружает данные из pickle файла.
    
    Args:
        filepath: Путь к файлу с данными
        
    Returns:
        DataFrame с данными об объектах недвижимости
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return pd.DataFrame(data)


def analyze_feature_importance(model, output_file: str = 'feature_importance.png', top_n: int = 30):
    """
    Анализирует важность признаков модели и создает визуализацию.
    
    Args:
        model: Обученная модель CatBoostRegressor
        output_file: Путь для сохранения графика
        top_n: Количество топ-признаков для визуализации
    """
    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("Важность признаков:")
    print(importance_df)
    
    importance_df.head(top_n).plot(x='feature', y='importance', kind='barh', figsize=(10, 12))
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"\nГрафик сохранен в {output_file}")


def main():
    """Основная функция для обучения модели и анализа признаков."""
    print("Загрузка данных...")
    df = delete_cols_with_any_nan(load_data())
    
    print(f"Загружено объектов: {len(df)}")
    print(f"Количество признаков: {len(df.columns)}")
    
    print("\nОбучение модели CatBoost...")
    model = train_model(df, PRICE_COL)
    
    print("\nАнализ важности признаков...")
    analyze_feature_importance(model)


if __name__ == '__main__':
    main()
