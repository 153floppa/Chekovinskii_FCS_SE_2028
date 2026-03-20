"""
🔗 Агрегатор данных: объединение продажи + аренда

Назначение:
    Объединяет две обработанные таблицы (продажи и аренды) в один единый датасет:
    1. Выбирает общие поля из обеих таблиц
    2. Добавляет поле Тип_сделки ('продажа' или 'аренда')
    3. Сохраняет в один JSON с единой структурой

Вход:
    - prodam_processed.json (обработанные продажи)
    - sdam_processed.json (обработанная аренда)

Выход:
    - merged.json (объединенный датасет)

Структура:
    Каждая запись содержит все общие поля (адрес, цена, координаты и т.д.)
    плюс новое поле Тип_сделки для различения источника.

Общие поля (31 поле):
    - Географические: Адрес, Город, Район, Регион, Широта, Долгота, Метро
    - Объявление: Заголовок, Описание, Ссылка на объявление, Дата объявления
    - Параметры: Категория, Подкатегория, Кол-во в комплекте
    - Продавец: Продавец, Контактное лицо, Тип продавца
    - Цена: Цена, Цена_за_кв_м, Цена_оригинальная, Конвертирована
    - Активность: Всего просмотров, Сегодня просмотров, Дата поднятия
    - Параметры: Параметры, Прайс-лист, Ссылки на картинки
    + Тип_сделки (добавлено)

Использование:
    python3 merge_datasets.py
"""
import json
from pathlib import Path

base_dir = Path(__file__).parent
prodam_file = base_dir / "prodam_processed.json"
sdam_file = base_dir / "sdam_processed.json"
output_file = base_dir / "merged.json"

# Загружаем оба файла
with open(prodam_file, 'r', encoding='utf-8') as f:
    prodam = json.load(f)['Sheet1']
with open(sdam_file, 'r', encoding='utf-8') as f:
    sdam = json.load(f)['Sheet1']

# Общие поля
common_fields = [
    'Адрес', 'Всего просмотров', 'Город', 'Дата объявления', 'Дата поднятия',
    'Долгота', 'Заголовок', 'Категория', 'Кол-во в комплекте', 'Конвертирована',
    'Контактное лицо', 'Метро', 'Метро. Время', 'Метро2', 'Номер объявления',
    'Описание', 'Параметры', 'Подкатегория', 'Прайс-лист', 'Продавец', 'Район',
    'Регион', 'Сегодня просмотров', 'Ссылка на объявление', 'Ссылки на картинки',
    'Тип продавца', 'Цена', 'Цена_за_кв_м', 'Цена_оригинальная', 'Широта', 'Этаж', 'Общая площадь'
]

merged = []

# Обрабатываем PRODAM
for row in prodam:
    new_row = {}
    for field in common_fields:
        new_row[field] = row.get(field, None)
    new_row['Тип_сделки'] = 'продажа'
    merged.append(new_row)

# Обрабатываем SDAM
for row in sdam:
    new_row = {}
    for field in common_fields:
        new_row[field] = row.get(field, None)
    new_row['Тип_сделки'] = 'аренда'
    merged.append(new_row)

# Сохраняем
output = {'Sheet1': merged}
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("✓ Объединение завершено!")
print(f"  PRODAM: {len(prodam)} записей")
print(f"  SDAM: {len(sdam)} записей")
print(f"  Всего: {len(merged)} записей")
print(f"\nПоля ({len(common_fields)}):")
for field in common_fields:
    print(f"  - {field}")
print(f"  - Тип_сделки (добавлено)")
