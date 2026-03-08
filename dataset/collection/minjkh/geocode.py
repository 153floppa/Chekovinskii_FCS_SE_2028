import json
import time
import requests
import os
from pathlib import Path

API_KEY = os.getenv("DADATA_API_KEY", "d18359aaa429cec57a609430d160110085c7d1fe")
SECRET_KEY = os.getenv("DADATA_SECRET_KEY", "d754b8f2a977b4f78e9f405903c6a826851ec0d5")
URL = "https://cleaner.dadata.ru/api/v1/clean/address"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": f"Token {API_KEY}",
    "X-Secret": SECRET_KEY,
}

DELAY = 0.06      # ~16 req/s — в рамках лимита 20 req/s


def geocode_one(address: str) -> dict:
    r = requests.post(URL, headers=HEADERS, json=[address], timeout=15)
    r.raise_for_status()
    return r.json()[0]


def main():
    base_dir = Path(__file__).parent
    input_file = base_dir / "houses.json"
    output_file = base_dir / "houses_filled.json"

    houses = json.load(open(input_file, encoding="utf-8"))

    # Подгружаем уже готовые результаты (возобновление после обрыва)
    try:
        done = json.load(open(output_file, encoding="utf-8"))
        done_set = {h["Адрес"] for h in done}
        print(f"Resuming: {len(done)} already geocoded")
    except FileNotFoundError:
        done = []
        done_set = set()

    pending = [h for h in houses if h["Адрес"] not in done_set]
    print(f"To geocode: {len(pending)}")

    SAVE_EVERY = 100

    for i, house in enumerate(pending):
        addr = f"Уфа, {house['Адрес']}"
        try:
            geo = geocode_one(addr)
            done.append({
                **house,
                "geo_lat": geo.get("geo_lat"),
                "geo_lon": geo.get("geo_lon"),
                "qc_geo": geo.get("qc_geo"),
            })
        except Exception as e:
            print(f"  Error [{i}] {addr}: {e} — skipping")
            done.append({**house, "geo_lat": None, "geo_lon": None, "qc_geo": None})

        if (i + 1) % SAVE_EVERY == 0:
            pct = (i + 1) / len(pending) * 100
            print(f"  {i + 1}/{len(pending)} ({pct:.1f}%)")
            # Only write updates, not the entire file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(done, f, ensure_ascii=False, indent=2, default=str)

        time.sleep(DELAY)

    # Final save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2, default=str)

    print(f"Done. Saved {len(done)} records to {output_file}")
    hits = sum(1 for h in done if h.get("qc_geo") == 0)
    print(f"Exact coordinates (qc_geo=0): {hits}/{len(done)}")


if __name__ == "__main__":
    main()
