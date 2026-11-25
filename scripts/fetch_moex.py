"""Выгрузка панелей MOEX (акции TQBR, облигации TQCB) через ISS API.

Запускается там, где резолвится iss.moex.com (VPS). Результат:
  shares.csv, bonds.csv, shares_ref.csv, bonds_ref.csv
"""

import concurrent.futures as cf
import csv
import json
import os
import ssl
import sys
import time
import urllib.request

BASE = "https://iss.moex.com/iss"
FROM = "2021-01-01"
TILL = "2026-11-25"
# отбор по обороту на старую и свежую дату — иначе в выборку попадают только
# недавно размещённые бумаги и обучающая часть оказывается почти пустой
LIQUID_DAYS = ("2023-06-01", "2026-11-24")
TOP_N = 80
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

HERE = os.path.dirname(os.path.abspath(__file__))
# iss.moex.com подписан цепочкой Russian Trusted (Минцифры), которой нет в системном
# хранилище — проверку не отключаем, а добавляем её корень явно.
CTX = ssl.create_default_context(cafile=os.path.join(HERE, "russian_trusted_ca.pem"))
HDRS = {"User-Agent": "Mozilla/5.0 (student research script)"}


def get(url, tries=4):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=40, context=CTX) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # сеть MOEX нестабильна
            last = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def block(d, name):
    cols = d[name]["columns"]
    return [dict(zip(cols, row)) for row in d[name]["data"]]


def day_turnover(market, board, date):
    """Оборот по всем бумагам за один день — для отбора ликвидных."""
    out, start = [], 0
    while True:
        url = (f"{BASE}/history/engines/stock/markets/{market}/boards/{board}"
               f"/securities.json?iss.meta=off&iss.only=history"
               f"&from={date}&till={date}&start={start}")
        rows = block(get(url), "history")
        out += rows
        if len(rows) < 100:
            return out
        start += len(rows)
        time.sleep(0.15)


def history(market, board, secid):
    rows, start = [], 0
    while True:
        url = (f"{BASE}/history/engines/stock/markets/{market}/boards/{board}"
               f"/securities/{secid}.json?iss.meta=off&iss.only=history"
               f"&from={FROM}&till={TILL}&start={start}")
        chunk = block(get(url), "history")
        rows += chunk
        if len(chunk) < 100:
            return rows
        start += len(chunk)
        time.sleep(0.15)


def ref(market, board, secid, cols):
    url = (f"{BASE}/engines/stock/markets/{market}/boards/{board}"
           f"/securities/{secid}.json?iss.meta=off&iss.only=securities"
           f"&securities.columns={cols}")
    rows = block(get(url), "securities")
    return rows[0] if rows else {}


def dump(name, rows):
    if not rows:
        print("!! пусто:", name)
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path = os.path.join(OUT, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} строк, {len(keys)} колонок")


def main():
    os.makedirs(OUT, exist_ok=True)

    only = sys.argv[2] if len(sys.argv) > 2 else None
    boards = [("shares", "TQBR"), ("bonds", "TQCB")]
    for market, board in boards:
        if only and only != market:
            continue
        scored = {}
        for day in LIQUID_DAYS:
            for r in day_turnover(market, board, day):
                sec = r.get("SECID")
                if not sec:
                    continue
                scored[sec] = scored.get(sec, 0.0) + float(r.get("VALUE") or 0)
        top = [s for s, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:TOP_N]]
        print(f"{board}: кандидатов {len(scored)}, берём {len(top)}", flush=True)

        def fetch(sec):
            return sec, history(market, board, sec)

        allrows, refs = [], []
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            for i, (sec, hist) in enumerate(pool.map(fetch, top), 1):
                if len(hist) < 250:  # слишком короткая история
                    print("  skip", sec, len(hist), flush=True)
                    continue
                allrows += hist
                if market == "shares":
                    refs.append(ref(market, board, sec, "SECID,SHORTNAME,SECNAME,LISTLEVEL,ISSUESIZE"))
                else:
                    refs.append(ref(market, board, sec,
                                    "SECID,SHORTNAME,SECNAME,LISTLEVEL,MATDATE,COUPONPERCENT,FACEVALUE,ISSUESIZE"))
                print(f"  {i}/{len(top)} {sec}: {len(hist)} строк", flush=True)

        dump(f"{market}.csv", allrows)
        dump(f"{market}_ref.csv", refs)


if __name__ == "__main__":
    main()
