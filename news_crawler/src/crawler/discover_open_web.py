"""Discover original publisher URLs without a fixed source allow-list."""
from __future__ import annotations

import csv
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlencode, urlsplit
from urllib.request import Request, urlopen

from lxml import html

from .discover_firecrawl import load_aliases, match_tickers
from .run_backfill import RAW_DATA_PATH, ROOT, _normalize_url

OUTPUT = ROOT / "data" / "staging" / "news_candidates_open_web.csv"
STATE = ROOT / "data" / "staging" / "open_web_discovery.sqlite"
FIELDS = ["source", "ticker", "url", "title", "description", "matched_text", "match_method", "match_confidence", "search_error"]


def _db():
    conn = sqlite3.connect(STATE)
    conn.execute("create table if not exists queries(query text primary key,status text,checked_at text default current_timestamp)")
    conn.execute("""create table if not exists candidates(
      source text,ticker text,url text,title text,description text,matched_text text,
      match_method text,match_confidence text,search_error text,
      primary key(ticker,url))""")
    return conn


def _known_urls():
    if not RAW_DATA_PATH.exists(): return set()
    with RAW_DATA_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return {_normalize_url(row["url"]) for row in csv.DictReader(handle)}


def _search(query: str):
    endpoint = "https://lite.duckduckgo.com/lite/?" + urlencode({"q": query})
    request = Request(endpoint, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    with urlopen(request, timeout=25) as response:
        doc = html.fromstring(response.read())
    results = []
    for anchor in doc.xpath("//a[@href]"):
        href = anchor.get("href", "")
        params = parse_qs(urlsplit(href).query)
        target = unquote(params.get("uddg", [""])[0])
        title = " ".join(anchor.itertext()).strip()
        if target.startswith(("http://", "https://")) and title:
            results.append((title, _normalize_url(target)))
    return results


def discover(delay=0.8, workers=2):
    aliases = load_aliases(); known = _known_urls(); conn = _db()
    queries = []
    for ticker, names in aliases.items():
        terms = [f'"{ticker}" cổ phiếu', *[f'"{name}"' for name in names]]
        for year in range(2020, 2027):
            queries.extend((ticker, f"{term} tin tức {year}") for term in terms)
    pending=[item for item in queries if not conn.execute("select 1 from queries where query=?",(item[1],)).fetchone()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0,len(pending),workers):
            batch=pending[start:start+workers]; futures=[pool.submit(_search,q) for _,q in batch]
            for (ticker,query),future in zip(batch,futures):
                try:
                    results=future.result(); status="ok" if results else "empty"
                except Exception as exc: results=[]; status=f"error:{type(exc).__name__}"
                for title,url in results:
                    if url in known: continue
                    matches=match_tickers(title,"",url,{ticker:aliases[ticker]})
                    if not matches: continue
                    match=matches[0]; host=(urlsplit(url).hostname or "").removeprefix("www.")
                    row={"source":host,"ticker":ticker,"url":url,"title":title,"description":"","matched_text":match["matched_text"],"match_method":match["match_method"],"match_confidence":match["match_confidence"],"search_error":""}
                    conn.execute(f"insert or ignore into candidates({','.join(FIELDS)}) values({','.join('?' for _ in FIELDS)})",tuple(row[k] for k in FIELDS))
                conn.execute("insert or replace into queries(query,status) values(?,?)",(query,status))
            conn.commit()
            if delay: time.sleep(delay)
    rows = conn.execute(f"select {','.join(FIELDS)} from candidates order by source,ticker,url").fetchall()
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(FIELDS); writer.writerows(rows)
    summary = {"queries": len(queries), "completed_queries": conn.execute("select count(*) from queries").fetchone()[0], "candidates": len(rows), "sources": conn.execute("select count(distinct source) from candidates").fetchone()[0]}
    conn.close(); print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": discover()
