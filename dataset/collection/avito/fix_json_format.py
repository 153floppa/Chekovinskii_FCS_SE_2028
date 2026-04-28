#!/usr/bin/env python3
"""
Переформатирует JSON в правильный формат для модели.
Основные изменения:
1. Цена → цена за м² (делим на площадь)
2. MIN расстояния → int (не float)
3. Добавляем ВСЕ POI поля со всеми радиусами
4. Правильный порядок полей
5. Координаты → null
"""

import json
import re
from pathlib import Path

# Все POI типы которые нужны (в правильном порядке)
POI_TYPES = [
    'Аптека', 'Автосервис', 'Пекарня', 'Банк', 'Бар', 'Быстропит',
    'Остановка', 'Чижик', 'Колледж', 'Детсад', 'Фитнес', 'Гимназия',
    'Кб', 'Кинотеатр', 'Лаборатория', 'Лицей', 'Магнит', 'МФЦ',
    'Перекресток', 'Пиццерия', 'Почта', 'ПВЗ', 'Пятерочка', 'Ресторан',
    'Школа', 'ВУЗ'
]

RADII = [50, 100, 150, 300, 500, 1000]

# Правильный порядок полей как в эталоне
FIELD_ORDER = [
    'Цена', 'Дата', 'Район', 'Тип', 'Источник', 'Этаж', 'Тип объявления',
    'Вид объекта', 'Общая площадь', 'Ссылка', 'Этажность здания',
]


def extract_area_from_params(params_str):
    """Извлекает площадь из поля Параметры если есть."""
    if not params_str or not isinstance(params_str, str):
        return None

    # Ищем "Общая площадь: XXX м2" или "Общая площадь, число: XXX"
    match = re.search(r'Общая площадь,?\s*(?:число)?:?\s*([\d,\.]+)', params_str)
    if match:
        area_str = match.group(1).replace(',', '.')
        try:
            return float(area_str)
        except (ValueError, TypeError):
            return None
    return None


def fix_object(obj):
    """Преобразует один объект в правильный формат."""
    fixed = {}

    # Получаем площадь - сначала пробуем из Общая площадь
    area = float(obj.get('Общая площадь', 0.0))

    # Если площадь нулевая, пробуем извлечь из Параметры
    if area == 0.0:
        area = extract_area_from_params(obj.get('Параметры'))
        if area is None:
            area = 1.0

    # На всякий случай еще раз проверяем
    if area == 0 or area is None:
        area = 1.0

    # Цена → цена за м²
    full_price = float(obj.get('Цена', 0.0))
    price_per_sqm = full_price / area if area > 0 else 0.0
    fixed['Цена'] = price_per_sqm

    # Базовые поля
    fixed['Дата'] = float(obj.get('Дата', 0.0))
    fixed['Район'] = str(obj.get('Район', ''))
    fixed['Тип'] = str(obj.get('Тип', 'Продам'))
    fixed['Источник'] = 'avito.ru'
    fixed['Этаж'] = int(obj.get('Этаж', 1))
    fixed['Тип объявления'] = 'Продам'
    fixed['Вид объекта'] = str(obj.get('Вид объекта', ''))
    fixed['Общая площадь'] = area  # Используем извлеченную площадь
    # Ссылка - пробуем разные имена колонок из Excel
    link = obj.get('Ссылка') or obj.get('Ссылка на объявление') or ''
    fixed['Ссылка'] = str(link) if link else ''
    fixed['Этажность здания'] = int(obj.get('Этажность здания', 1))

    # Жилфонд
    fixed['Квартир200'] = int(obj.get('Квартир200', 0))
    fixed['Квартир500'] = int(obj.get('Квартир500', 0))
    fixed['Площадь200'] = float(obj.get('Площадь200', 0.0))
    fixed['Площадь500'] = float(obj.get('Площадь500', 0.0))
    fixed['Год200'] = int(obj.get('Год200', 2000))
    fixed['Год500'] = int(obj.get('Год500', 2000))
    fixed['Домов200'] = int(obj.get('Домов200', 0))
    fixed['Домов500'] = int(obj.get('Домов500', 0))

    # POI - добавляем ВСЕ поля
    for poi_type in POI_TYPES:
        # Добавляем все радиусы
        for radius in RADII:
            field_name = f"{poi_type}{radius}"
            fixed[field_name] = int(obj.get(field_name, 0))

        # Добавляем MIN (как INT!)
        field_min = f"{poi_type}MIN"
        min_val = obj.get(field_min, 0)
        if isinstance(min_val, float):
            fixed[field_min] = int(round(min_val))
        else:
            fixed[field_min] = int(min_val) if min_val else 0

    # Координаты → null
    fixed['Долгота'] = None
    fixed['Широта'] = None

    # Среднее и медиана соседей (если есть)
    fixed['avg500'] = int(obj.get('avg500', 0)) if obj.get('avg500') else 0
    fixed['avg1000'] = int(obj.get('avg1000', 0)) if obj.get('avg1000') else 0
    fixed['med500'] = int(obj.get('med500', 0)) if obj.get('med500') else 0
    fixed['med1000'] = int(obj.get('med1000', 0)) if obj.get('med1000') else 0

    # Цена оригинальная и коэффициент
    fixed['Цена_оригинальная'] = float(full_price)
    fixed['Коэффициент_индексации'] = 1.28

    # Метаданные
    fixed['source'] = 'original'
    fixed['source_date'] = '2024-04-01'
    fixed['source_description'] = 'Исходный датасет (result4, индексировано)'

    return fixed


def fix_json_file(input_path, output_path):
    """Преобразует весь JSON файл."""
    print(f"\n{'='*80}")
    print(f"FIXING JSON FORMAT")
    print(f"{'='*80}\n")

    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"  ✓ Loaded {len(data)} objects")

    print(f"\nTransforming objects...")
    fixed_data = []
    for idx, obj in enumerate(data):
        if idx % 100 == 0:
            print(f"  → Processing {idx}/{len(data)}...")
        fixed_obj = fix_object(obj)
        fixed_data.append(fixed_obj)

    print(f"  ✓ Transformed {len(fixed_data)} objects")

    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved successfully")

    print(f"\n{'='*80}")
    print(f"✅ TRANSFORMATION COMPLETE")
    print(f"{'='*80}\n")

    # Show sample
    print("Sample object (first):")
    print(json.dumps(fixed_data[0], ensure_ascii=False, indent=2)[:500] + "...\n")


if __name__ == '__main__':
    input_file = 'convert/test_sample_50.json'
    output_file = 'convert/test_sample_50_fixed.json'

    fix_json_file(input_file, output_file)
