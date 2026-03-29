"""
Скрипт для обогащения объектов из avito/merged.json информацией о POI.
Добавляет к каждому объекту информацию о ближайших магазинах, школах, банках и т.д.

Использование:
    python3 enrich_merged_objects.py

Результат:
    dataset/data/dataset_final.json — датасет с добавленной POI информацией
"""

import json
import math
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расчет расстояния между двумя точками в метрах (формула Haversine)."""
    R = 6371000  # Радиус Земли в метрах

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def get_poi_near_object(
    obj: Dict,
    poi_by_type: Dict[str, List[Dict]],
    lat_field: str = 'Широта',
    lon_field: str = 'Долгота'
) -> Dict:
    """
    Находит все POI рядом с объектом и возвращает статистику.

    Args:
        obj: объект из merged.json с координатами
        poi_by_type: словарь POI, предварительно сгруппированный по типам (для оптимизации)
        lat_field: название поля широты в объекте
        lon_field: название поля долготы в объекте

    Returns:
        Словарь с информацией о POI в разных радиусах
    """

    # Проверяем наличие координат
    try:
        obj_lat = float(obj.get(lat_field))
        obj_lon = float(obj.get(lon_field))
    except (ValueError, TypeError, AttributeError):
        return {}

    if obj_lat is None or obj_lon is None:
        return {}

    # Радиусы для подсчета
    RADIUS_THRESHOLDS = [50, 100, 150, 300, 500, 1000]  # метры

    result = {}

    # Для каждого типа POI считаем статистику
    for poi_type, poi_list in poi_by_type.items():
        # Считаем расстояния до всех POI этого типа
        distances = []
        poi_by_radius = {r: [] for r in RADIUS_THRESHOLDS}

        for poi in poi_list:
            try:
                poi_lat = float(poi.get('lat'))
                poi_lon = float(poi.get('lon'))
            except (ValueError, TypeError, AttributeError):
                continue

            distance = haversine_distance(obj_lat, obj_lon, poi_lat, poi_lon)
            distances.append(distance)

            # Добавляем в соответствующие радиусы
            for radius in RADIUS_THRESHOLDS:
                if distance <= radius:
                    poi_by_radius[radius].append(poi)

        if not distances:
            continue

        # MIN — минимальное расстояние до этого типа POI
        min_distance = min(distances)
        result[f'{poi_type}MIN'] = round(min_distance, 1)

        # Для каждого радиуса — кол-во POI
        for radius in RADIUS_THRESHOLDS:
            count = len(poi_by_radius[radius])
            result[f'{poi_type}{radius}'] = count

    return result


def enrich_single_object(obj: Dict, poi_by_type: Dict[str, List[Dict]]) -> Dict:
    """
    Обогащает один объект информацией о POI.

    Args:
        obj: объект из merged.json
        poi_by_type: словарь POI, предварительно сгруппированный по типам

    Returns:
        Обогащенный объект со всеми исходными полями + POI информация
    """
    enriched = obj.copy()

    # Получаем информацию о POI
    poi_info = get_poi_near_object(obj, poi_by_type)

    # Добавляем в объект
    enriched.update(poi_info)

    return enriched


def enrich_merged_json(
    input_path: str = None,
    poi_path: str = None,
    output_path: str = None
) -> None:
    """
    Главная функция: читает merged.json, обогащает каждый объект POI информацией,
    сохраняет результат в новый файл.
    """

    # Установка путей по умолчанию
    if input_path is None:
        input_path = str(Path(__file__).parent.parent / 'avito' / 'merged.json')
    if poi_path is None:
        poi_path = str(Path(__file__).parent / 'ufa_poi.json')
    if output_path is None:
        output_path = str(Path(__file__).parent.parent.parent / 'data' / 'dataset_final.json')

    print("=" * 80)
    print("ОБОГАЩЕНИЕ MERGED.JSON ИНФОРМАЦИЕЙ О POI")
    print("=" * 80)

    # Загружаем POI данные
    print(f"\n→ Загрузка POI данных из {poi_path}...")
    try:
        with open(poi_path, 'r', encoding='utf-8') as f:
            poi_data = json.load(f)
        print(f"✓ Загружено {len(poi_data)} POI объектов")
    except FileNotFoundError:
        print(f"✗ Файл не найден: {poi_path}")
        return
    except json.JSONDecodeError:
        print(f"✗ Ошибка при чтении JSON: {poi_path}")
        return

    # Загружаем merged.json
    print(f"\n→ Загрузка объектов из {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Проверяем структуру
        if isinstance(data, dict) and 'Sheet1' in data:
            objects = data['Sheet1']
        elif isinstance(data, list):
            objects = data
        else:
            print(f"✗ Неожиданная структура файла")
            return

        print(f"✓ Загружено {len(objects)} объектов из merged.json")
    except FileNotFoundError:
        print(f"✗ Файл не найден: {input_path}")
        return
    except json.JSONDecodeError:
        print(f"✗ Ошибка при чтении JSON: {input_path}")
        return

    # Группируем POI по типам один раз (для оптимизации)
    poi_by_type = {}
    for poi in poi_data:
        poi_type = poi.get('type')
        if poi_type not in poi_by_type:
            poi_by_type[poi_type] = []
        poi_by_type[poi_type].append(poi)

    # Обогащаем каждый объект
    print(f"\n→ Обогащение объектов POI информацией...")
    enriched_objects = []
    start_time = time.time()

    for i, obj in enumerate(objects):
        if i % 100 == 0:
            print(f"  Обработано {i}/{len(objects)}...")

        enriched = enrich_single_object(obj, poi_by_type)
        enriched_objects.append(enriched)

    elapsed = time.time() - start_time
    print(f"✓ Обогащено {len(enriched_objects)} объектов за {elapsed:.1f}с ({elapsed/len(enriched_objects)*1000:.2f}мс на объект)")

    # Сохраняем результат
    print(f"\n→ Сохранение результата в {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_objects, f, ensure_ascii=False, indent=2)

    print(f"✓ Сохранено {len(enriched_objects)} объектов")

    # Статистика
    print(f"\n" + "=" * 80)
    print(f"СТАТИСТИКА")
    print(f"=" * 80)

    if enriched_objects:
        first_obj = enriched_objects[0]

        # Считаем POI поля
        poi_fields = [k for k in first_obj.keys() if any(
            cat in k for cat in [
                'Аптека', 'Автосервис', 'Пекарня', 'Банк', 'Бар', 'Быстропит',
                'Остановка', 'Колледж', 'Детсад', 'Фитнес', 'Кинотеатр',
                'Лаборатория', 'Лицей', 'Магнит', 'ПВЗ', 'Пятерочка', 'Ресторан',
                'Школа', 'ВУЗ', 'Чижик', 'Гимназия', 'МФЦ', 'Перекресток', 'Почта', 'Клуб'
            ]
        )]

        print(f"Всего полей в объекте: {len(first_obj)}")
        print(f"Добавлено POI полей: {len(poi_fields)}")
        print(f"Исходных полей: {len(first_obj) - len(poi_fields)}")

        # Примеры
        print(f"\nПримеры POI полей (первый объект):")
        sample_fields = sorted([k for k in poi_fields if 'MIN' in k])[:5]
        for field in sample_fields:
            value = first_obj.get(field)
            print(f"  {field}: {value}")

        print(f"\nПримеры радиус-полей (первый объект):")
        sample_fields = [k for k in poi_fields if k.endswith('500') or k.endswith('1000')][:5]
        for field in sample_fields:
            value = first_obj.get(field)
            print(f"  {field}: {value}")

    print(f"\n✓ Готово! Результат в {output_path}")


if __name__ == '__main__':
    enrich_merged_json()
