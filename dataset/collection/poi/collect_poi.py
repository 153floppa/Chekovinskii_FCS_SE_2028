"""
Сбор данных POI (Points of Interest) для Уфы через Nominatim API (OpenStreetMap).
Данные используются для обогащения датасета коммерческой недвижимости.

API: https://nominatim.openstreetmap.org/
Лицензия данных: ODbL (OpenStreetMap contributors)
"""

import json
import time
import logging
import requests
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Nominatim требует идентификатор приложения в User-Agent
HEADERS = {
    'User-Agent': 'UfaRealEstatePOI/1.0 (prodimgame@gmail.com)'
}

BASE_URL = 'https://nominatim.openstreetmap.org/search'

# Уфа — ограничивающий прямоугольник (viewbox): W,S,E,N
UFA_VIEWBOX = '55.6,54.55,56.35,54.95'

# Категории POI с их Nominatim-тегами и русскими названиями
POI_CATEGORIES = {
    'Аптека':      {'amenity': 'pharmacy'},
    'Автосервис':  {'amenity': 'car_repair'},
    'Пекарня':     {'shop': 'bakery'},
    'Банк':        {'amenity': 'bank'},
    'Бар':         {'amenity': 'bar'},
    'Быстропит':   {'amenity': 'fast_food'},
    'Остановка':   {'highway': 'bus_stop'},
    'Колледж':     {'amenity': 'college'},
    'Детсад':      {'amenity': 'kindergarten'},
    'Фитнес':      {'leisure': 'fitness_centre'},
    'Кинотеатр':   {'amenity': 'cinema'},
    'Лаборатория': {'amenity': 'clinic'},
    'Лицей':       {'amenity': 'school', 'name': 'лицей'},
    'Гимназия':    {'amenity': 'school', 'name': 'гимназия'},
    'Школа':       {'amenity': 'school'},
    'ВУЗ':         {'amenity': 'university'},
    'Ресторан':    {'amenity': 'restaurant'},
    'МФЦ':         {'amenity': 'government'},
    'Почта':       {'amenity': 'post_office'},
    'Клуб':        {'amenity': 'nightclub'},
    'Фитнес':      {'leisure': 'sports_centre'},
}

# Сетевые магазины — ищем по названию
CHAIN_STORES = {
    'Магнит':      'Магнит',
    'Пятерочка':   'Пятёрочка',
    'Перекресток': 'Перекрёсток',
    'Чижик':       'Чижик',
    'ПВЗ':         'СДЭК',
}

# Задержка между запросами — Nominatim требует не более 1 req/sec
REQUEST_DELAY = 1.1


def fetch_poi(query: str, category_tag: Optional[dict] = None, limit: int = 50) -> list:
    """
    Запрашивает POI у Nominatim с пагинацией (все результаты).

    Args:
        query: текстовый запрос (для сетей)
        category_tag: словарь тег→значение для поиска по OSM-тегам
        limit: результатов за один запрос (max 50)

    Returns:
        Список всех объектов из API (с пролистыванием)
    """
    all_results = []
    offset = 0
    max_iterations = 50  # безопасность от бесконечного цикла

    for iteration in range(max_iterations):
        params = {
            'format': 'jsonv2',
            'addressdetails': 0,
            'extratags': 1,
            'limit': limit,
            'offset': offset,  # пагинация
            'viewbox': UFA_VIEWBOX,
            'bounded': 1,
            'countrycodes': 'ru',
        }

        if category_tag:
            tag_key = list(category_tag.keys())[0]
            tag_val = category_tag[tag_key]
            params['q'] = f'[{tag_key}={tag_val}]'
        else:
            params['q'] = query

        try:
            resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            batch = resp.json()
        except requests.RequestException as e:
            log.warning(f'Ошибка запроса (смещение {offset}): {e}')
            break

        if not batch:
            # Нет больше результатов
            break

        all_results.extend(batch)
        log.debug(f'    Смещение {offset}: получено {len(batch)} объектов')

        # Если получили меньше чем лимит, это последняя страница
        if len(batch) < limit:
            break

        offset += limit
        time.sleep(REQUEST_DELAY)

    return all_results


def parse_result(raw: dict, type_label: str) -> Optional[dict]:
    """Приводит сырой ответ Nominatim к нужной структуре."""
    try:
        lat = float(raw['lat'])
        lon = float(raw['lon'])
    except (KeyError, ValueError):
        return None

    # Фильтруем объекты за пределами Уфы
    if not (54.55 <= lat <= 54.95 and 55.6 <= lon <= 56.35):
        return None

    return {
        'type': type_label,
        'name': raw.get('display_name', '').split(',')[0].strip(),
        'address': raw.get('display_name', ''),
        'lat': lat,
        'lon': lon,
        'osm_type': raw.get('osm_type', ''),
        'osm_id': raw.get('osm_id', 0),
        'source': 'OpenStreetMap/Nominatim',
    }


def collect_all_poi(output_path: str = None) -> list:
    """
    Основная функция сбора всех POI.
    Обходит все категории и сохраняет результат в JSON.
    """
    if output_path is None:
        output_path = str(Path(__file__).parent / 'ufa_poi.json')

    results = []
    seen_osm_ids = set()

    def add_results(raw_list: list, label: str):
        added = 0
        for raw in raw_list:
            osm_id = raw.get('osm_id')
            if osm_id and osm_id in seen_osm_ids:
                continue
            parsed = parse_result(raw, label)
            if parsed:
                results.append(parsed)
                if osm_id:
                    seen_osm_ids.add(osm_id)
                added += 1
        log.info(f'  → добавлено {added} объектов (дедупликация по osm_id)')

    # Сбор по тегам
    log.info('=== Сбор POI по категориям ===')
    for label, tags in POI_CATEGORIES.items():
        log.info(f'Запрос: {label} ({tags})')

        # Для Лицея/Гимназии дополнительно фильтруем по имени
        name_filter = tags.pop('name', None)
        raw = fetch_poi(query='', category_tag=tags)

        if name_filter:
            raw = [r for r in raw
                   if name_filter.lower() in r.get('display_name', '').lower()]
            tags['name'] = name_filter  # возвращаем обратно

        add_results(raw, label)
        time.sleep(REQUEST_DELAY)

    # Сбор сетевых магазинов по названию
    log.info('\n=== Сбор сетевых магазинов ===')
    for label, query in CHAIN_STORES.items():
        log.info(f'Запрос: {label} ("{query}")')
        raw = fetch_poi(query=query)
        add_results(raw, label)
        time.sleep(REQUEST_DELAY)

    # Сохранение
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log.info(f'\n✓ Сохранено {len(results)} объектов → {output_path}')
    return results


def print_stats(data: list):
    """Выводит статистику по типам собранных объектов."""
    from collections import Counter
    counts = Counter(obj['type'] for obj in data)
    print(f'\n{"Тип POI":<20} {"Количество":>10}')
    print('-' * 32)
    for poi_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f'{poi_type:<20} {count:>10}')
    print(f'{"ИТОГО":<20} {len(data):>10}')


if __name__ == '__main__':
    log.info('Запуск сбора POI для Уфы (Nominatim/OSM)')
    log.info(f'Регион: viewbox={UFA_VIEWBOX}')
    log.info(f'Категорий: {len(POI_CATEGORIES) + len(CHAIN_STORES)}\n')

    poi_data = collect_all_poi()
    print_stats(poi_data)
