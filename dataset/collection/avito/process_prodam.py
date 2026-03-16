"""
💰 Обработчик данных ПРОДАЖИ

Назначение:
    Нормализует данные о продажах недвижимости:
    1. Извлекает площадь из поля "Параметры" (через регулярное выражение)
    2. Вычисляет цену за кв. метр: цена / площадь
    3. Фильтрует выбросы (цена/м² < 30k или > 600k руб/м²)
    4. Сохраняет оригинальную цену для аудита

Вход:
    - prodam.json (парсед из Excel)

Выход:
    - prodam_processed.json (готов к объединению)

Логика фильтрации:
    ❌ Нет цены                          → удаляется
    ❌ Нет площади                       → удаляется
    ❌ Неверная цена (не число)          → удаляется
    ❌ Неверная площадь                  → удаляется
    ❌ Цена/м² < 30,000 руб              → выброс (слишком дешево)
    ❌ Цена/м² > 600,000 руб             → выброс (слишком дорого)
    ✅ Остальное                          → сохраняется

Новые поля:
    - Цена_за_кв_м (вычисленная)
    - Цена_оригинальная (исходная цена)
    - Площадь (извлеченная)
    - Конвертирована (True = обработана)

Использование:
    python3 process_prodam.py
"""

import json
import re
from pathlib import Path

def extract_area_from_params(params_str):
    """Извлекает площадь из строки Параметры"""
    if not params_str:
        return None

    # Ищем "Общая площадь: XXX м2" или "Общая площадь: XXX,XXX м2"
    match = re.search(r'Общая площадь:\s*([\d,\.]+)\s*м', params_str)
    if match:
        area_str = match.group(1).replace(',', '.')
        try:
            return float(area_str)
        except:
            return None
    return None

def process_prodam_data():
    """Обрабатывает данные продажи - переводит в цену за кв. м"""

    base_dir = Path(__file__).parent
    input_file = base_dir / "prodam.json"
    output_file = base_dir / "prodam_processed.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    skipped = {
        'no_price': 0,
        'no_area': 0,
        'invalid_price': 0,
        'invalid_area': 0,
        'out_of_range': 0,
    }

    for row in data['Sheet1']:
        try:
            price = row.get('Цена')
            params = row.get('Параметры')

            if price is None:
                skipped['no_price'] += 1
                continue

            # Извлекаем площадь из параметров
            area_value = extract_area_from_params(params)

            if area_value is None:
                skipped['no_area'] += 1
                continue

            try:
                price_value = float(price) if price is not None else None
            except (ValueError, TypeError):
                skipped['invalid_price'] += 1
                continue

            if price_value is None or price_value <= 0:
                skipped['invalid_price'] += 1
                continue

            if area_value <= 0:
                skipped['invalid_area'] += 1
                continue

            # Вычисляем цену за кв. м
            price_per_sqm = price_value / area_value

            # Проверяем диапазон
            if price_per_sqm < 30000 or price_per_sqm > 600000:
                skipped['out_of_range'] += 1
                continue

            # Обновляем цену в записи
            row['Цена'] = price_per_sqm
            row['Цена_оригинальная'] = price
            row['Цена_за_кв_м'] = price_per_sqm
            row['Общая площадь'] = area_value
            row['Конвертирована'] = True

            processed_data.append(row)

        except Exception as e:
            continue

    output_data = {'Sheet1': processed_data}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("✓ Обработка prodam завершена!")
    print(f"  Обработано: {len(processed_data)} / {len(data['Sheet1'])}")
    print(f"  Удалено: {len(data['Sheet1']) - len(processed_data)}")

    print(f"\nПричины удаления:")
    print(f"  - Нет цены: {skipped['no_price']}")
    print(f"  - Нет площади: {skipped['no_area']}")
    print(f"  - Неверная цена: {skipped['invalid_price']}")
    print(f"  - Неверная площадь: {skipped['invalid_area']}")
    print(f"  - Вне диапазона (< 30000 или > 600000): {skipped['out_of_range']}")

    if processed_data:
        prices = [row['Цена_за_кв_м'] for row in processed_data]
        print(f"\n  Цены за кв. м: {min(prices):,.0f} - {max(prices):,.0f}")
        print(f"  Средняя цена: {sum(prices) / len(prices):,.0f}")

if __name__ == '__main__':
    process_prodam_data()
