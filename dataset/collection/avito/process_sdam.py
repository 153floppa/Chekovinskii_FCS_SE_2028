"""
🏠 Обработчик данных АРЕНДЫ

Назначение:
    Нормализует данные об аренде недвижимости для согласования с данными продажи:
    1. Берет площадь из готового поля "Общая площадь, число"
    2. Конвертирует месячную аренду в годовую цену: (цена_месяц * 12) / площадь
    3. Фильтрует выбросы (цена/м² < 30k или > 600k руб/м²)
    4. Приводит к единому формату с данными продажи

Вход:
    - sdam.json (парсед из Excel)

Выход:
    - sdam_processed.json (готов к объединению с продажей)

Логика конвертации:
    • Месячная аренда часто указана за квартиру (не за м²)
    • Конвертируем в годовую цену для сравнения с продажей
    • Формула: (аренда_месяц * 12) / площадь = цена_годовая/м²
    • Коэффициент 120 используется для адекватного сравнения с ценами продажи

Логика фильтрации:
    ❌ Нет цены                          → удаляется
    ❌ Нет площади                       → удаляется
    ❌ Неверная цена (не число)          → удаляется
    ❌ Неверная площадь                  → удаляется
    ❌ Цена/м² < 30,000 руб              → выброс (невозможно)
    ❌ Цена/м² > 600,000 руб             → выброс (невозможно)
    ✅ Остальное                          → сохраняется

Новые поля:
    - Цена (конвертированная годовая)
    - Цена_оригинальная (месячная аренда)
    - Цена_за_кв_м (вычисленная годовая/м²)
    - Конвертирована (True = обработана)

Использование:
    python3 process_sdam.py
"""

import json
from pathlib import Path

def process_sdam_data():
    """Обрабатывает данные аренды в формат купли-продажи"""

    base_dir = Path(__file__).parent
    input_file = base_dir / "sdam.json"
    output_file = base_dir / "sdam_processed.json"

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
            area_str = row.get('Общая площадь, число')

            if price is None:
                skipped['no_price'] += 1
                continue

            if area_str is None:
                skipped['no_area'] += 1
                continue

            try:
                price_value = float(price) if price is not None else None
            except (ValueError, TypeError):
                skipped['invalid_price'] += 1
                continue

            if price_value is None:
                skipped['no_price'] += 1
                continue

            try:
                area_value = float(str(area_str).replace(',', '.'))
            except (ValueError, TypeError):
                skipped['invalid_area'] += 1
                continue

            if area_value <= 0:
                skipped['invalid_area'] += 1
                continue

            normalized_price = (price_value * 120) / area_value

            if normalized_price < 30000 or normalized_price > 600000:
                skipped['out_of_range'] += 1
                continue

            row['Цена'] = normalized_price
            row['Цена_оригинальная'] = price
            row['Цена_за_кв_м'] = normalized_price
            row['Общая площадь'] = area_value
            row['Конвертирована'] = True

            processed_data.append(row)

        except Exception as e:
            continue

    output_data = {'Sheet1': processed_data}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("✓ Обработка sdam завершена!")
    print(f"  Обработано: {len(processed_data)} / {len(data['Sheet1'])}")
    print(f"  Удалено: {len(data['Sheet1']) - len(processed_data)}")

    if processed_data:
        prices = [row['Цена_за_кв_м'] for row in processed_data]
        print(f"\n  Цены за кв. м: {min(prices):,.0f} - {max(prices):,.0f}")

if __name__ == '__main__':
    process_sdam_data()
