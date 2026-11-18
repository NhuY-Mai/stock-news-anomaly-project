"""Scrape discovered candidates and emit verified article/ticker datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

from dateutil import parser as date_parser

from .discover_firecrawl import CANDIDATES_PATH, FIRECRAWL_DIR, _firecrawl_command, load_aliases, match_tickers
from .run_backfill import ROOT, _normalize_url
from .store import merge_into_dataset

ARTICLE_CACHE = FIRECRAWL_DIR / "articles"
SCRAPE_REPORT_PATH = ROOT / "outputs" / "coverage" / "firecrawl_scrape_report.csv"
_RATE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


def _respect_rate_limit(interval_seconds: float = 6.5) -> None:
    """Keep requests below Firecrawl's 10 requests/minute account limit."""
    global _LAST_REQUEST_AT
    with _RATE_LOCK:
        wait = interval_seconds - (time.monotonic() - _LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()


def _cache_path(url: str) -> Path:
    return ARTICLE_CACHE / f"{hashlib.sha256(url.encode()).hexdigest()}.json"


def _scrape_one(url: str) -> tuple[str, Path, str]:
    ARTICLE_CACHE.mkdir(parents=True, exist_ok=True)
    output = _cache_path(url)
    if output.exists() and output.stat().st_size > 100:
        return url, output, "cached"
    command = _firecrawl_command("scrape", url, "--format", "markdown,links", "--only-main-content", "-o", str(output), "--json")
    last_error = "scrape failed"
    for attempt in range(3):
        _respect_rate_limit()
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
        if completed.returncode == 0 and output.exists():
            return url, output, "ok"
        last_error = (completed.stderr or completed.stdout or last_error).strip()
        if "Rate limit exceeded" not in last_error:
            break
        time.sleep(10 * (attempt + 1))
    return url, output, last_error


def _plain_text(markdown: str) -> str:
    text = re.sub(r"!\[[^]]*]\([^)]*\)", " ", markdown)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`#>*_|~-]+", " ", text)
    return " ".join(text.split())


def _metadata_value(metadata: dict, *keys: str) -> str:
    for key in keys:
        if metadata.get(key):
            return " ".join(str(metadata[key]).split())
    return ""


def _publication_date(metadata: dict, markdown: str) -> tuple[str, str]:
    raw = _metadata_value(metadata, "article:published_time", "publishedTime", "datePublished", "publicationDate")
    if not raw:
        match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](20\d{2})\b", markdown[:2000])
        raw = match.group(0) if match else ""
    if not raw:
        return "", ""
    try:
        return date_parser.parse(raw, dayfirst=True, fuzzy=True).date().isoformat(), raw
    except (ValueError, OverflowError):
        return "", raw


def _evidence_for(ticker: str, title: str, summary: str, content: str, aliases: dict[str, list[str]]) -> dict | None:
    matches = match_tickers(title, summary, "", {ticker: aliases[ticker]})
    if matches:
        return matches[0]
    # The first section is used to avoid matches coming only from unrelated recommendation widgets.
    matches = match_tickers(title, content[:6000], "", {ticker: aliases[ticker]})
    return matches[0] if matches else None


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def scrape_candidates(start: date, end: date, workers: int = 2, max_per_ticker_source: int = 0) -> tuple[list[dict], list[dict]]:
    aliases = load_aliases()
    with CANDIDATES_PATH.open(encoding="utf-8-sig", newline="") as handle:
        discovered = [row for row in csv.DictReader(handle) if row.get("ticker") and row.get("url")]
    pair_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    candidates: list[dict] = []
    for row in discovered:
        key = (row["source"], row["ticker"])
        if max_per_ticker_source > 0 and pair_counts[key] >= max_per_ticker_source:
            continue
        pair_counts[key] += 1
        candidates.append(row)
    by_url: dict[str, list[dict]] = {}
    for row in candidates:
        by_url.setdefault(row["url"], []).append(row)
    scrape_status: dict[str, tuple[Path, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 2))) as pool:
        futures = {pool.submit(_scrape_one, url): url for url in by_url}
        for future in as_completed(futures):
            url, path, status = future.result()
            scrape_status[url] = (path, status)

    articles_by_url: dict[str, dict] = {}
    links: list[dict] = []
    reports: list[dict] = []
    crawl_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for discovered_url, associations in by_url.items():
        path, status = scrape_status[discovered_url]
        if status not in {"ok", "cached"}:
            reports.append({"url": discovered_url, "status": "scrape_error", "error_message": status})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            markdown = payload.get("markdown", "")
            content = _plain_text(markdown)
            title = _metadata_value(metadata, "og:title", "ogTitle", "title") or associations[0]["title"]
            summary = _metadata_value(metadata, "og:description", "ogDescription", "description") or associations[0]["description"]
            published_date, published_raw = _publication_date(metadata, markdown)
            canonical = _normalize_url(_metadata_value(metadata, "og:url", "ogUrl", "url", "sourceURL") or discovered_url)
            if len(title) < 8 or len(content) < 200 or not published_date:
                raise ValueError("missing title, publication date, or meaningful article content")
            published = date.fromisoformat(published_date)
            if not start <= published <= end:
                raise ValueError("publication date outside stock-data range")
            evidence: list[tuple[dict, dict]] = []
            for association in associations:
                match = _evidence_for(association["ticker"], title, summary, content, aliases)
                if match:
                    evidence.append((association, match))
            if not evidence:
                raise ValueError("no ticker/company evidence in scraped article")
            news_id = hashlib.sha256(canonical.encode()).hexdigest()
            article = {
                "news_id": news_id, "source": associations[0]["source"], "url": canonical,
                "title": title, "summary": summary, "content": content,
                "published_date": published_date, "published_at_raw": published_raw,
                "author": _metadata_value(metadata, "article:author", "author"),
                "category": _metadata_value(metadata, "article:section", "section"),
                "crawl_time": crawl_time, "crawl_status": "ok", "error_message": "",
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
            }
            articles_by_url[canonical] = article
            for association, match in evidence:
                links.append({"news_id": news_id, "ticker": association["ticker"], "match_method": match["match_method"], "match_confidence": match["match_confidence"], "matched_text": match["matched_text"], "published_date": published_date, "source": association["source"], "url": canonical})
            reports.append({"url": discovered_url, "status": "ok", "error_message": ""})
        except Exception as exc:
            reports.append({"url": discovered_url, "status": "rejected", "error_message": str(exc)})

    articles = sorted(articles_by_url.values(), key=lambda row: (row["published_date"], row["source"], row["url"]))
    unique_links = {(row["news_id"], row["ticker"]): row for row in links}
    link_rows = sorted(unique_links.values(), key=lambda row: (row["published_date"], row["ticker"], row["news_id"]))
    # Append to the existing datasets instead of overwriting them.
    merge_into_dataset(articles, link_rows)
    _write_csv(SCRAPE_REPORT_PATH, reports, ["url", "status", "error_message"])
    return articles, link_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-09-25")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-per-ticker-source", type=int, default=0, help="0 means unlimited")
    args = parser.parse_args()
    articles, links = scrape_candidates(date.fromisoformat(args.start_date), date.fromisoformat(args.end_date), args.workers, args.max_per_ticker_source)
    print(json.dumps({"verified_articles": len(articles), "ticker_links": len(links), "sources": len({row['source'] for row in articles}), "tickers": len({row['ticker'] for row in links})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
