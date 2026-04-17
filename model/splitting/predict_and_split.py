"""
Модуль для предсказания цен и оптимального разбиения объектов недвижимости.

Принимает датасет входных данных (в формате до очистки), предсказывает стоимость
с помощью обученной модели, рассчитывает оптимальное дробление для максимизации
стоимости и сохраняет результаты в JSON для отображения в app.py.

Сохраняет оригинальные данные объекта (включая Ссылку) для аналитики.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import pickle

from model.regressors.catboost_regressor import get_model
from model.splitting.generate_split import split
from interface.utils import delete_cols_with_any_nan, get_columns_grouped_by_dtypes, x_y_split
from config import PRICE_COL, SQUARE_COL


# Столбцы, которые нужно сохранить для отображения в app.py
DISPLAY_COLUMNS = [
    'Ссылка',          # для открытия в браузере
    'Район',           # локация
    'Вид объекта',     # тип
    'Этаж',
    'Этажность здания',
]


def predict_prices(df: pd.DataFrame, model) -> np.ndarray:
    """Предсказывает цены для датасета."""
    X, _ = x_y_split(df, PRICE_COL)
    cat_cols = list(get_columns_grouped_by_dtypes(df)['object'])

    from catboost import Pool
    pool = Pool(X, cat_features=cat_cols)
    price_per_sqm = model.predict(pool)
    full_price = price_per_sqm * df[SQUARE_COL]
    return full_price


def calculate_optimal_split(
    obj: pd.Series,
    original_obj: pd.Series,
    model,
    cat_cols: List[str],
    obj_idx: int
) -> Dict[str, Any]:
    """
    Рассчитывает оптимальное разбиение для одного объекта.

    Args:
        obj: Series с числовыми признаками (для модели)
        original_obj: Series с оригинальными данными (для отображения)
        model: Обученная модель CatBoost
        cat_cols: Список названий категориальных признаков
        obj_idx: Индекс объекта

    Returns:
        Словарь с информацией о разбиении объекта
    """
    try:
        max_price, split_objects = split(obj, model, cat_cols)

        # Основная информация
        # Переводим цену за м² в полную цену для сравнения
        original_full_price = obj[PRICE_COL] * obj[SQUARE_COL]

        result = {
            'object_id': obj_idx,
            'link': original_obj.get('Ссылка', ''),
            'district': original_obj.get('Район', 'N/A'),
            'object_type': original_obj.get('Вид объекта', 'N/A'),
            'floor': original_obj.get('Этаж', 'N/A'),
            'building_floors': original_obj.get('Этажность здания', 'N/A'),
            'original_price': float(original_full_price),
            'original_square': float(obj[SQUARE_COL]),
            'optimal_price': float(max_price),
            'profit_increase': float(max_price - original_full_price),
            'profit_increase_pct': float((max_price - original_full_price) / original_full_price * 100) if original_full_price > 0 else 0,
            'should_split': max_price > original_full_price,
            'num_parts': len(split_objects),
            'parts': []
        }

        # Информация о частях после разбиения
        for i, part in enumerate(split_objects):
            part_dict = {
                'part_id': i + 1,
                'square': float(part[SQUARE_COL]),
                'predicted_price': float(part[PRICE_COL]),
                'price_per_sqm': float(part[PRICE_COL] / part[SQUARE_COL]) if part[SQUARE_COL] > 0 else 0,
            }
            result['parts'].append(part_dict)

        return result

    except Exception as e:
        original_full_price = obj[PRICE_COL] * obj[SQUARE_COL]
        return {
            'object_id': obj_idx,
            'link': original_obj.get('Ссылка', ''),
            'original_price': float(original_full_price),
            'original_square': float(obj[SQUARE_COL]),
            'error': str(e),
            'should_split': False
        }


def process_analytics_dataset(
    input_path: str,
    output_path: str = 'results/analytics.json'
) -> Dict[str, Any]:
    """
    Обрабатывает датасет для аналитики: предсказывает цены и разбиение.

    Функция работает с оригинальным форматом данных (до очистки) и сохраняет
    всю необходимую информацию для отображения в app.py.

    Args:
        input_path: Путь к входному датасету (JSON или PKL, формат dataset_final.json)
        output_path: Путь для сохранения результатов

    Returns:
        Словарь со статистикой обработки
    """
    print(f"\n📂 Загрузка данных из {input_path}...")

    # Загружаем данные
    if input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df_original = pd.DataFrame(data)
    elif input_path.endswith('.pkl'):
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        df_original = pd.DataFrame(data)
    else:
        raise ValueError("Поддерживаются только JSON и PKL форматы")

    print(f"✅ Загружено: {len(df_original)} объектов")

    # Очищаем данные для модели (удаляем NaN и служебные столбцы)
    print("🧹 Очистка данных...")
    df_clean = delete_cols_with_any_nan(df_original)
    print(f"✅ После очистки: {len(df_clean)} объектов")

    if len(df_clean) == 0:
        raise ValueError("После очистки не осталось данных!")

    # Сохраняем оригинальные данные для отображения перед удалением столбцов
    df_display = df_clean.copy()

    # Оставляем только числовые столбцы и только те, которые были в обучающих данных
    print("🔢 Фильтруем колонки под обученную модель...")
    with open('dataset/data/result.pkl', 'rb') as f:
        training_data = pickle.load(f)
    training_cols = list(pd.DataFrame(training_data).columns)

    # Берем пересечение: есть ли обучающие колонки в нашем датасете
    available_cols = [c for c in training_cols if c in df_clean.columns]
    df_clean = df_clean[available_cols]
    print(f"✅ Осталось {len(df_clean.columns)} колонок (из {len(training_cols)} обучающих)")

    # Загружаем модель
    print("🤖 Загрузка модели...")
    model = get_model('catboost_regressor')
    print("✅ Модель загружена")

    # После удаления категориальных столбцов их уже нет
    cat_cols = []  # пусто, так как все категориальные уже удалены

    # Обрабатываем каждый объект
    print("🔄 Расчет оптимального разбиения...")
    split_results = []

    for idx, (idx_clean, row_clean) in enumerate(df_clean.iterrows()):
        # Берем оригинальные данные по индексу (для ссылок и отображения)
        if idx_clean in df_display.index:
            row_original = df_display.loc[idx_clean]
        else:
            row_original = row_clean

        # Рассчитываем разбиение
        result = calculate_optimal_split(row_clean, row_original, model, [], idx)
        split_results.append(result)

        # Выводим статус каждого объекта
        if 'error' in result:
            print(f"  ❌ Объект {idx}: ОШИБКА - {result['error']}")
        else:
            status = "✅ Выгодно разбить" if result.get('should_split') else "⚪ Не разбивать"
            print(f"  {status} - Объект {idx} ({result.get('district', '?')})")

        if (idx + 1) % 100 == 0:
            print(f"  ⏳ Обработано {idx + 1}/{len(df_clean)}")

    print(f"✅ Обработано {len(split_results)} объектов")

    # Сохраняем результаты
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Конвертируем numpy типы в Python native для JSON
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    split_results = [convert_to_serializable(r) for r in split_results]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(split_results, f, ensure_ascii=False, indent=2)

    print(f"💾 Результаты сохранены в {output_file}")

    # Статистика
    objects_to_split = sum(1 for r in split_results if r.get('should_split', False))
    errors = sum(1 for r in split_results if 'error' in r)
    total_profit = sum(r.get('profit_increase', 0) for r in split_results if 'profit_increase' in r)
    avg_profit_pct = np.mean([r.get('profit_increase_pct', 0) for r in split_results if 'profit_increase_pct' in r])

    stats = {
        'status': '✅ Успешно',
        'total_objects': len(split_results),
        'objects_to_split': objects_to_split,
        'split_percentage': f"{objects_to_split / len(split_results) * 100:.1f}%",
        'errors': errors,
        'total_profit_increase': f"₽{total_profit:,.0f}",
        'average_profit_increase_pct': f"{avg_profit_pct:.1f}%",
        'output_file': str(output_file)
    }

    print("\n" + "="*70)
    print("📊 СТАТИСТИКА АНАЛИТИКИ")
    print("="*70)
    for key, value in stats.items():
        if key != 'output_file':
            print(f"  {key}: {value}")
    print(f"  Результаты: {stats['output_file']}")
    print("="*70 + "\n")

    return stats


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'results/analytics.json'
        process_analytics_dataset(input_file, output_file)
    else:
        print("📊 Модуль для предсказания и разбиения недвижимости\n")
        print("Использование: python predict_and_split.py <input_file> [output_file]\n")
        print("Параметры:")
        print("  input_file  - датасет в формате JSON или PKL")
        print("  output_file - файл для результатов (опционально)\n")
        print("Примеры:")
        print("  python predict_and_split.py data/dataset_final.json")
        print("  python predict_and_split.py data/analytics_sample.pkl results/analytics.json")
