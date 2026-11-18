"""Merge verified Thao rows with canonical news without modifying either raw file."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.crawler.run_backfill import ROOT

MAIN = ROOT / "data" / "raw" / "news_raw.csv"
LINKS = ROOT / "data" / "final" / "news_ticker.csv"
THAO = Path(r"C:\Users\DELL\Downloads\DATA_THAO - NEWS_RAW.csv")
OUTPUT = ROOT / "data" / "final" / "News_raw.csv"
REPORT = ROOT / "outputs" / "coverage" / "merge_thao_report.json"
START, END = "2020-01-01", "2026-09-25"
TICKERS = {"ANV", "DBC", "DGW", "DHG", "FRT", "IMP", "MSN", "MWG", "PNJ", "QNS", "SAB", "SBT", "VHC", "VNM"}
FIELDS = [
    "news_id", "source", "url", "title", "summary", "content", "published_date",
    "published_at_raw", "author", "category", "crawl_time", "crawl_status",
    "content_hash", "industry_group", "tickers", "keywords",
    "event_type", "note", "dataset_origin",
    "source_news_ids", "source_content_hashes", "original_crawl_status",
]


def read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_key(url: str):
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                       if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}])
    host = parts.netloc.lower().removeprefix("www.")
    return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/") or "/", query, ""))


def split_values(value: str, allowed=None):
    values = [part.strip() for part in (value or "").split(",") if part.strip()]
    if allowed is not None: values = [value.upper() for value in values if value.upper() in allowed]
    return values


def join_unique(values):
    return ", ".join(dict.fromkeys(value for value in values if value))


def main():
    main_rows, link_rows, thao_rows = read(MAIN), read(LINKS), read(THAO)
    linked = defaultdict(list)
    for row in link_rows: linked[row["news_id"]].append(row["ticker"])

    merged = {}
    for row in main_rows:
        out = {field: row.get(field, "") for field in FIELDS}
        out.update({
            "tickers": join_unique(linked[row["news_id"]]),
            "crawl_status": "success",
            "dataset_origin": "main_crawler",
            "source_news_ids": row["news_id"],
            "source_content_hashes": row["content_hash"],
            "original_crawl_status": "success",
        })
        merged[canonical_key(row["url"])] = out

    rejected = defaultdict(int); accepted = defaultdict(list)
    for row in thao_rows:
        if row.get("crawl_status") != "success": rejected["not_success"] += 1; continue
        if not row.get("content", "").strip(): rejected["missing_content"] += 1; continue
        if not START <= row.get("published_date", "") <= END: rejected["outside_date_range"] += 1; continue
        accepted[canonical_key(row["url"])].append(row)

    added = enriched = internal_duplicates = 0
    for key, group in accepted.items():
        internal_duplicates += len(group) - 1
        best = max(group, key=lambda row: len(row.get("content", "")))
        group_tickers = join_unique(t for row in group for t in split_values(row.get("tickers", ""), TICKERS))
        group_keywords = join_unique(x for row in group for x in split_values(row.get("keywords", "")))
        group_events = join_unique(row.get("event_type", "").strip() for row in group)
        group_ids = join_unique(row.get("news_id", "") for row in group)
        group_hashes = join_unique(row.get("content_hash", "") for row in group)
        if key in merged:
            out = merged[key]
            out["tickers"] = join_unique([*split_values(out["tickers"]), *split_values(group_tickers)])
            for field, value in (("industry_group", best.get("industry_group", "")), ("keywords", group_keywords), ("event_type", group_events), ("note", best.get("note", ""))):
                if not out.get(field) and value: out[field] = value
            out["dataset_origin"] = "main_crawler|thao_crawler"
            out["source_news_ids"] = join_unique([out["source_news_ids"], group_ids])
            out["source_content_hashes"] = join_unique([out["source_content_hashes"], group_hashes])
            out["original_crawl_status"] = "success"
            enriched += 1
            continue

        url = key
        content = best["content"].strip()
        out = {field: best.get(field, "") for field in FIELDS}
        out.update({
            "news_id": hashlib.sha256(url.encode()).hexdigest(),
            "url": url,
            "crawl_status": "success",
            "content_hash": hashlib.sha256(" ".join(content.split()).encode()).hexdigest(),
            "tickers": group_tickers,
            "keywords": group_keywords,
            "event_type": group_events,
            "dataset_origin": "thao_crawler",
            "source_news_ids": group_ids,
            "source_content_hashes": group_hashes,
            "original_crawl_status": "success",
        })
        merged[key] = out; added += 1

    rows = sorted(merged.values(), key=lambda row: (row["published_date"], row["source"], row["url"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    report = {
        "main_articles": len(main_rows), "thao_input_rows": len(thao_rows),
        "thao_valid_rows": sum(len(group) for group in accepted.values()),
        "thao_valid_unique_urls": len(accepted), "thao_internal_duplicate_rows": internal_duplicates,
        "thao_articles_added": added, "existing_articles_enriched": enriched,
        "rejected": dict(rejected), "output_articles": len(rows),
        "output_unique_urls": len({canonical_key(row["url"]) for row in rows}),
        "output_unique_news_ids": len({row["news_id"] for row in rows}),
        "output_unique_content_hashes": len({row["content_hash"] for row in rows}),
        "date_range": [min(row["published_date"] for row in rows), max(row["published_date"] for row in rows)],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
