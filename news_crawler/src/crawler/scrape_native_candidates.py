"""Fetch Firecrawl-discovered URLs directly and append only verified articles."""
from __future__ import annotations
import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from lxml import html

from .discover_firecrawl import load_aliases, match_tickers
from .run_backfill import RAW_DATA_PATH, ROOT, _canonical_url, _extract_author, _extract_category, _extract_content, _extract_published_date, _extract_summary, _extract_title, _normalize_url, _parse_date, _safe_fetch, _write_csv
from .store import merge_into_dataset


def _fetch(url: str):
    try:
        return url, html.fromstring(_safe_fetch(url, timeout=15, retries=2, delay=1)), ""
    except Exception as exc:
        return url, None, str(exc)


def append_candidates(candidate_path: Path, start: date, end: date, workers: int = 4):
    aliases = load_aliases()
    with candidate_path.open(encoding="utf-8-sig", newline="") as handle:
        candidates = [r for r in csv.DictReader(handle) if r.get("ticker") and r.get("url")]
    by_url = {}
    for row in candidates:
        by_url.setdefault(row["url"], []).append(row)
    known_urls = set()
    if RAW_DATA_PATH.exists():
        with RAW_DATA_PATH.open(encoding="utf-8-sig", newline="") as handle:
            known_urls = {r["url"] for r in csv.DictReader(handle)}
    fetched = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_fetch, url): url for url in by_url if _normalize_url(url) not in known_urls}
        for future in as_completed(futures):
            url, doc, error = future.result()
            fetched[url] = (doc, error)
    new_news, new_links, report = [], [], []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for url, associations in by_url.items():
        if _normalize_url(url) in known_urls:
            continue
        doc, error = fetched.get(url, (None, "not fetched"))
        if doc is None:
            report.append({"url": url, "status": "fetch_error", "error_message": error}); continue
        source = associations[0]
        title, summary, content = _extract_title(doc), _extract_summary(doc), _extract_content(doc)
        raw_date = _extract_published_date(doc); parsed = _parse_date(raw_date)
        canonical = _canonical_url(doc, url, source["url"].split('/')[2].removeprefix('www.'))
        if len(title) < 8 or len(content) < 200 or not parsed or not start <= parsed <= end:
            report.append({"url": url, "status": "rejected", "error_message": "missing title/content/date or date outside range"}); continue
        evidence = []
        for association in associations:
            matches = match_tickers(title, f"{summary} {content[:6000]}", canonical, {association["ticker"]: aliases[association["ticker"]]})
            if matches: evidence.append((association, matches[0]))
        if not evidence:
            report.append({"url": url, "status": "rejected", "error_message": "no ticker/company evidence"}); continue
        news_id = hashlib.sha256(canonical.encode()).hexdigest()
        article_text = " ".join((title, summary, content))
        new_news.append({"news_id": news_id, "source": source["source"], "url": canonical, "title": title, "summary": summary, "content": content, "published_date": parsed.isoformat(), "published_at_raw": raw_date, "author": _extract_author(doc), "category": _extract_category(doc), "crawl_time": now, "crawl_status": "ok", "error_message": "", "content_hash": hashlib.sha256(" ".join(article_text.split()).encode()).hexdigest()})
        for association, match in evidence:
            new_links.append({"news_id": news_id, "ticker": association["ticker"], "match_method": match["match_method"], "match_confidence": match["match_confidence"], "matched_text": match["matched_text"], "published_date": parsed.isoformat(), "source": source["source"], "url": canonical})
        report.append({"url": url, "status": "ok", "error_message": ""})
    merge_into_dataset(new_news, new_links)
    _write_csv(ROOT / "outputs" / "coverage" / "native_more_report.csv", report, ["url", "status", "error_message"])
    return new_news, new_links


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--candidates", type=Path, required=True); parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(); news, links = append_candidates(args.candidates, date(2020,1,1), date(2026,9,25), args.workers)
    print({"new_articles": len(news), "new_links": len(links)})


if __name__ == "__main__": main()
