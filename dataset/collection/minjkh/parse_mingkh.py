import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://dom.mingkh.ru/bashkortostan/ufa/houses"
PAGES = 56
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9",
}

CAPTCHA_MARKERS = ("captcha_answer", "Хотим убедиться, что вы не робот")
# Match expression only inside the label text, e.g. "Сколько будет 9 - 1?"
LABEL_RE = re.compile(r"Сколько будет\s+(-?\d+)\s*([+\-*])\s*(-?\d+)\s*\?", re.IGNORECASE)


def is_captcha(html: str) -> bool:
    return any(m in html for m in CAPTCHA_MARKERS)


def solve_captcha(session: requests.Session, url: str, html: str) -> str:
    m = LABEL_RE.search(html)
    if not m:
        raise RuntimeError(f"captcha expression not found in label. snippet: {html[html.find('Сколько'):html.find('Сколько')+80]!r}")
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    answer = {"+": a + b, "-": a - b, "*": a * b}[op]
    print(f"[captcha] {a} {op} {b} = {answer}")
    # POST to base URL without ?page=N — captcha belongs to the session, not the page
    base_url = url.split("?")[0]
    r = session.post(url, data={"captcha_answer": str(answer)},
                     headers={"Referer": url}, allow_redirects=True)
    r.raise_for_status()
    if is_captcha(r.text):
        # Server gave another captcha — either wrong answer or new session needed
        print(f"[captcha] POST response snippet: {r.text[:400]!r}")
        raise RuntimeError("captcha still present after submit")
    return r.text


def fetch(session: requests.Session, url: str) -> str:
    r = session.get(url)
    r.raise_for_status()
    if is_captcha(r.text):
        return solve_captcha(session, url, r.text)
    return r.text


def parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = None
    for t in soup.find_all("table"):
        if t.find("tbody") and t.find("tbody").find("tr"):
            table = t
            break
    if table is None:
        return []

    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        values = [td.get_text(strip=True) for td in cells]
        row = dict(zip(headers, values)) if headers else {f"col_{i}": v for i, v in enumerate(values)}
        link = tr.find("a", href=re.compile(r"/house/"))
        if link:
            row["url"] = "https://dom.mingkh.ru" + link["href"]
        rows.append(row)
    return rows


def main() -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows: list[dict] = []
    for page in range(1, PAGES + 1):
        url = f"{BASE}?page={page}"
        html = fetch(session, url)
        rows = parse_page(html)
        all_rows.extend(rows)
        print(f"page {page}: +{len(rows)} (total {len(all_rows)})")
        time.sleep(1)

    output_file = Path(__file__).parent / "houses.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(f"saved {len(all_rows)} rows -> {output_file}")


if __name__ == "__main__":
    main()
