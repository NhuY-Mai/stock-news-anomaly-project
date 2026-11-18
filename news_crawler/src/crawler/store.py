"""Append verified articles to the canonical datasets without losing existing rows."""
from __future__ import annotations

import csv
from pathlib import Path

from .run_backfill import FIELDNAMES, RAW_DATA_PATH, ROOT, _write_csv

LINKS_PATH = ROOT / "data" / "final" / "news_ticker.csv"
LINK_FIELDS = ["news_id", "ticker", "match_method", "match_confidence", "matched_text", "published_date", "source", "url"]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_into_dataset(new_news: list[dict], new_links: list[dict], news_path: Path = RAW_DATA_PATH, links_path: Path = LINKS_PATH) -> tuple[list[dict], list[dict]]:
    """Merge new rows into news_raw/news_ticker. Existing rows always win.

    Articles are de-duplicated by URL, then by content_hash; ticker links of a
    dropped duplicate are moved to the article that is kept.
    """
    existing_news, existing_links = _read(news_path), _read(links_path)
    news_map: dict[str, dict] = {}
    for article in [*existing_news, *new_news]:
        news_map.setdefault(article["url"], article)
    hash_map: dict[str, dict] = {}
    id_remap: dict[str, dict] = {}
    unique_news: list[dict] = []
    for article in news_map.values():
        kept = hash_map.get(article["content_hash"])
        if kept:
            id_remap[article["news_id"]] = kept
        else:
            hash_map[article["content_hash"]] = article
            unique_news.append(article)
    link_map: dict[tuple[str, str], dict] = {}
    for link in [*existing_links, *new_links]:
        if link["news_id"] in id_remap:
            kept = id_remap[link["news_id"]]
            link = {**link, "news_id": kept["news_id"], "source": kept["source"], "url": kept["url"], "published_date": kept["published_date"]}
        link_map.setdefault((link["news_id"], link["ticker"]), link)
    news_rows = sorted(unique_news, key=lambda r: (r["published_date"], r["source"]))
    link_rows = sorted(link_map.values(), key=lambda r: (r["published_date"], r["ticker"]))
    _write_csv(news_path, news_rows, FIELDNAMES)
    _write_csv(links_path, link_rows, LINK_FIELDS)
    return news_rows, link_rows
