"""
Модуль подготовки датасета для обучения модели.

Загружает raw датасет, очищает его от ненужных столбцов и пропусков,
сохраняет готовый датасет и обучает модель для проверки готовности.
"""

import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PRICE_COL, SQUARE_COL
from interface.utils import delete_cols_with_any_nan, get_columns_grouped_by_dtypes, x_y_split
from model.regressors.catboost_regressor import train_model


COLUMNS_TO_REMOVE = [
    'Ссылка',
    'Источник',
    'Тип',
    'Тип объявления',
    'Цена_оригинальная',      # Удаляем для чистоты модели
    'Коэффициент_индексации', # Коррелирует с целевой переменной
]


def load_raw_dataset(json_path: str) -> pd.DataFrame:
    """Загружает raw датасет из JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет ненужные столбцы и все категориальные (string/object) столбцы."""
    cols_to_drop = [col for col in COLUMNS_TO_REMOVE if col in df.columns]
    df_cleaned = df.drop(columns=cols_to_drop)
    print(f"Удалено столбцов: {len(cols_to_drop)}")
    print(f"Удаленные столбцы: {cols_to_drop}")

    # Удаляем категориальные столбцы (string, object) - оставляем только числовые
    cat_cols = [col for col in df_cleaned.columns if col != PRICE_COL and df_cleaned[col].dtype in ('object', 'string')]
    if cat_cols:
        df_cleaned = df_cleaned.drop(columns=cat_cols)
        print(f"Удалено категориальных столбцов: {len(cat_cols)} {cat_cols[:5]}")

    return df_cleaned


def remove_columns_with_missing_values(df: pd.DataFrame, min_non_null_ratio: float = 0.8) -> pd.DataFrame:
    """
    Удаляет столбцы с большим количеством пропусков.

    Args:
        df: DataFrame для обработки
        min_non_null_ratio: Минимальная доля непропущенных значений для сохранения столбца
    """
    initial_cols = len(df.columns)
    df_clean = df.dropna(axis=1, thresh=len(df) * min_non_null_ratio)
    removed_cols = initial_cols - len(df_clean.columns)
    print(f"Удалено столбцов с пропусками (< {min_non_null_ratio*100:.0f}% непропущенных): {removed_cols}")

    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    removed_rows = initial_rows - len(df_clean)
    print(f"Удалено строк с оставшимися пропусками: {removed_rows} из {initial_rows}")

    return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет дубликаты строк."""
    initial_rows = len(df)
    df_clean = df.drop_duplicates()
    removed_rows = initial_rows - len(df_clean)
    print(f"Удалено дубликатов: {removed_rows}")
    return df_clean


def convert_and_clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразует типы данных и удаляет столбцы с некорректными типами."""
    df_clean = df.copy()

    for col in df_clean.columns:
        if col == PRICE_COL:
            continue

        # Конвертируем pandas string в обычный object (str)
        if df_clean[col].dtype.name == 'string':
            df_clean[col] = df_clean[col].astype('object')

        # Если столбец помечен как object, но большинство значений числовые - преобразуем
        if df_clean[col].dtype == 'object':
            try:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                non_null_before = df_clean[col].notna().sum()
                df_clean = df_clean.dropna(subset=[col])
                if len(df_clean) == 0:
                    raise ValueError("Все строки были удалены при конвертации")
            except:
                # Если не удалось преобразовать, оставляем как есть (категориальная переменная)
                pass

    return df_clean


def get_categorical_columns(df: pd.DataFrame) -> list:
    """Определяет категориальные столбцы (string и object)."""
    categorical_dtypes = ('string', 'object', 'category')
    return [col for col in df.columns if df[col].dtype.name in categorical_dtypes or df[col].dtype == 'object']


def validate_dataset(df: pd.DataFrame) -> bool:
    """Проверяет, что датасет содержит необходимые столбцы."""
    required_cols = [PRICE_COL, SQUARE_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют необходимые столбцы: {missing}")

    if (df[PRICE_COL] <= 0).any() or (df[SQUARE_COL] <= 0).any():
        raise ValueError("Найдены отрицательные или нулевые значения в цене или площади")

    return True


def prepare_dataset(
    raw_json_path: str = 'dataset/data/dataset_final.json',
    output_dir: str = 'dataset/aggregation/prepared_data'
) -> Tuple[pd.DataFrame, str]:
    """
    Подготавливает датасет для обучения модели.

    Args:
        raw_json_path: Путь к исходному JSON датасету
        output_dir: Папка для сохранения готовых данных

    Returns:
        Кортеж (очищенный DataFrame, путь к сохраненному файлу)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ПОДГОТОВКА ДАТАСЕТА")
    print("=" * 80)

    print("\n1️⃣  Загрузка raw датасета...")
    df = load_raw_dataset(raw_json_path)
    print(f"   Загружено строк: {len(df)}")
    print(f"   Загружено столбцов: {len(df.columns)}")

    print("\n2️⃣  Удаление ненужных столбцов...")
    df = remove_unnecessary_columns(df)
    print(f"   Осталось столбцов: {len(df.columns)}")

    print("\n3️⃣  Удаление столбцов и строк с пропусками...")
    df = remove_columns_with_missing_values(df)
    print(f"   Осталось столбцов: {len(df.columns)}")
    print(f"   Осталось строк: {len(df)}")

    print("\n4️⃣  Удаление дубликатов...")
    df = remove_duplicates(df)
    print(f"   Осталось строк: {len(df)}")

    print("\n5️⃣  Преобразование типов данных...")
    df = convert_and_clean_types(df)
    print(f"   Осталось строк: {len(df)}")

    print("\n6️⃣  Валидация датасета...")
    validate_dataset(df)
    print("   ✅ Датасет прошел валидацию")

    print("\n7️⃣  Сохранение очищенного датасета...")
    output_file = output_path / 'dataset_prepared.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(df.to_dict('records'), f, ensure_ascii=False, indent=2)
    print(f"   Сохранено: {output_file}")

    # Подготовим данные для pickle с конвертацией типов
    df_for_pickle = df.copy()
    for col in df_for_pickle.columns:
        if df_for_pickle[col].dtype.name == 'string':
            df_for_pickle[col] = df_for_pickle[col].astype('object')
        elif df_for_pickle[col].dtype == 'object':
            # Убедимся, что object столбцы содержат обычные Python типы
            df_for_pickle[col] = df_for_pickle[col].astype(str)

    data_for_pickle = df_for_pickle.to_dict('records')

    pickle_file = output_path / 'dataset_prepared.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(data_for_pickle, f)
    print(f"   Сохранено: {pickle_file}")

    # Также сохраняем в dataset/data/result.pkl для совместимости с model/training/main.py
    result_pkl = Path('dataset/data/result.pkl')
    with open(result_pkl, 'wb') as f:
        pickle.dump(data_for_pickle, f)
    print(f"   Сохранено (для main.py): {result_pkl}")

    print("\n8️⃣  Обучение модели (маркер готовности)...")
    try:
        from catboost import CatBoostRegressor
        from sklearn.model_selection import train_test_split
        from config import RANDOM_SEED

        X, y = x_y_split(df, PRICE_COL)

        # Оставляем только числовые столбцы (избегаем проблем с типами данных)
        X = X.select_dtypes(include=[np.number])

        # Убедимся, что нет пропусков
        X = X.dropna()
        y = y[X.index]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        model = CatBoostRegressor(
            iterations=50,
            verbose=False,
            random_seed=RANDOM_SEED,
            loss_function='RMSE'
        )

        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"   ✅ Модель успешно обучена (R²={test_score:.4f})!")

        model_file = output_path / 'training_model'
        model.save_model(str(model_file))
        print(f"   Модель сохранена: {model_file}")
    except Exception as e:
        raise Exception(f"❌ Ошибка при обучении модели: {e}")

    print("\n" + "=" * 80)
    print("ПОДГОТОВКА ЗАВЕРШЕНА УСПЕШНО")
    print("=" * 80)
    print(f"\nИтоговая статистика:")
    print(f"  Строк: {len(df)}")
    print(f"  Столбцов: {len(df.columns)}")
    print(f"  Столбцы: {list(df.columns)}")
    print(f"\nСохраненные файлы в {output_dir}:")
    print(f"  - dataset_prepared.json")
    print(f"  - dataset_prepared.pkl")
    print(f"  - training_model")
    print(f"\nТакже сохранено в dataset/data/result.pkl для совместимости с model/training/main.py")

    return df, str(output_file)


if __name__ == '__main__':
    df, output_path = prepare_dataset()
