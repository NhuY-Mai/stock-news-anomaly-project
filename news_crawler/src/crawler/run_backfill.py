"""Crawl verified articles; never synthesize missing news data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import ssl
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib import error, request
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from dateutil import parser as date_parser
from lxml import etree, html

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "project_config.json"
SOURCES_PATH = ROOT / "config" / "sources.csv"
RAW_DATA_PATH = ROOT / "data" / "raw" / "news_raw.csv"
# Seed-page backfill does not verify ticker relevance, so it never writes the canonical dataset.
BACKFILL_PATH = ROOT / "data" / "staging" / "news_backfill.csv"
REPORT_PATH = ROOT / "outputs" / "coverage" / "crawl_report.csv"
LOG_PATH = ROOT / "logs" / "crawl.log"
FIELDNAMES = ["news_id", "source", "url", "title", "summary", "content", "published_date", "published_at_raw", "author", "category", "crawl_time", "crawl_status", "error_message", "content_hash"]
TRACKING_PARAMS = {"fbclid", "gclid", "utm_campaign", "utm_content", "utm_medium", "utm_source", "utm_term"}
SKIP_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf", ".mp4", ".mp3", ".zip")


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _read_sources() -> list[dict]:
    with SOURCES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if str(row.get("enabled", "")).lower() == "true"]


def _safe_fetch(url: str, timeout: int = 10, retries: int = 1, delay: float = 0.5) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.6",
    }
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            try:
                import certifi
                context = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                context = ssl.create_default_context()
            with request.urlopen(request.Request(url, headers=headers), timeout=timeout, context=context) as response:
                content_type = response.headers.get_content_type()
                if content_type not in {"text/html", "application/xhtml+xml", "application/xml", "text/xml"}:
                    raise ValueError(f"unsupported content type: {content_type}")
                payload = response.read()
                charset = response.headers.get_content_charset()
                if not charset:
                    match = re.search(br"charset\s*=\s*['\"]?([\w.-]+)", payload[:4096], re.I)
                    charset = match.group(1).decode("ascii", "ignore") if match else "utf-8"
                return payload.decode(charset, errors="replace")
        except (error.HTTPError, error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if isinstance(exc, error.HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                break
            if attempt + 1 < retries:
                time.sleep(delay * (2**attempt))
    raise RuntimeError(str(last_error or "fetch failed"))


def _normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode([(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key.lower() not in TRACKING_PARAMS])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", query, ""))


def _same_domain(url: str, domain: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    expected = domain.lower().removeprefix("www.")
    return host == expected or host.endswith("." + expected)


def _text_value(doc, *selectors: str) -> str:
    for selector in selectors:
        values = doc.xpath(selector)
        if values:
            value = values[0]
            if isinstance(value, etree._Element):
                value = " ".join(value.itertext())
            value = " ".join(str(value).split())
            if value:
                return value
    return ""


def _json_ld(doc) -> list[dict]:
    records: list[dict] = []
    for raw in doc.xpath('//script[@type="application/ld+json"]/text()'):
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        queue = value if isinstance(value, list) else [value]
        for item in queue:
            if isinstance(item, dict) and isinstance(item.get("@graph"), list):
                queue.extend(item["@graph"])
            if isinstance(item, dict):
                records.append(item)
    return records


def _ld_value(doc, *keys: str) -> str:
    for record in _json_ld(doc):
        kinds = record.get("@type", "")
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if not any(str(kind).lower() in {"article", "newsarticle", "reportagenewsarticle"} for kind in kinds):
            continue
        for key in keys:
            value = record.get(key)
            if isinstance(value, dict):
                value = value.get("name")
            if isinstance(value, list):
                value = ", ".join(str(item.get("name", "") if isinstance(item, dict) else item) for item in value)
            if value:
                return " ".join(str(value).split())
    return ""


def _extract_title(doc) -> str:
    return _text_value(doc, '//meta[@property="og:title"]/@content', '//meta[@name="twitter:title"]/@content', "//h1") or _ld_value(doc, "headline", "name")


def _extract_summary(doc) -> str:
    return _text_value(doc, '//meta[@name="description"]/@content', '//meta[@property="og:description"]/@content', '//*[contains(@class,"sapo") or contains(@class,"summary") or contains(@class,"description")][1]') or _ld_value(doc, "description")


def _extract_content(doc) -> str:
    selectors = [
        '//article//*[self::p or self::h2]/descendant-or-self::*/text()',
        '//*[contains(@class,"detail-content") or contains(@class,"article-content") or contains(@class,"news-content") or contains(@class,"content-detail")]//*[self::p or self::h2]/descendant-or-self::*/text()',
        '//main//*[self::p or self::h2]/descendant-or-self::*/text()',
    ]
    for selector in selectors:
        pieces = [" ".join(str(piece).split()) for piece in doc.xpath(selector)]
        text = " ".join(piece for piece in pieces if piece)
        if len(text) >= 120:
            return text
    return _ld_value(doc, "articleBody")


def _extract_published_date(doc) -> str:
    value = _text_value(doc, '//meta[@property="article:published_time"]/@content', '//meta[@name="pubdate"]/@content', '//meta[@name="publishdate"]/@content', '//*[@itemprop="datePublished"]/@content', "//time/@datetime") or _ld_value(doc, "datePublished")
    if value:
        return value
    visible = _text_value(doc, '//*[contains(@class,"date") or contains(@class,"time") or contains(@class,"publish")][1]')
    match = re.search(r"(?:(\d{1,2}):?(\d{2})?\s+)?(\d{1,2})[/-](\d{1,2})[/-](20\d{2})", visible)
    return match.group(0) if match else ""


def _extract_author(doc) -> str:
    return _text_value(doc, '//meta[@name="author"]/@content', '//meta[@property="article:author"]/@content', '//*[@itemprop="author"]/@content', '//*[contains(@class,"author")][1]') or _ld_value(doc, "author")


def _extract_category(doc) -> str:
    return _text_value(doc, '//meta[@property="article:section"]/@content', '//meta[@name="category"]/@content') or _ld_value(doc, "articleSection")


def _canonical_url(doc, fallback: str, domain: str) -> str:
    canonical = _text_value(doc, '//link[contains(concat(" ",normalize-space(@rel)," ")," canonical ")]/@href', '//meta[@property="og:url"]/@content')
    candidate = _normalize_url(urljoin(fallback, canonical)) if canonical else _normalize_url(fallback)
    return candidate if _same_domain(candidate, domain) else _normalize_url(fallback)


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return date_parser.parse(value, dayfirst=True, fuzzy=True).date()
    except (ValueError, OverflowError):
        return None


def _standardize_row(source: dict, article_url: str, doc) -> dict:
    url = _canonical_url(doc, article_url, source["domain"])
    title, summary, content = _extract_title(doc), _extract_summary(doc), _extract_content(doc)
    published_at_raw = _extract_published_date(doc)
    published = _parse_date(published_at_raw)
    article_text = " ".join(part for part in (title, summary, content) if part)
    return {
        "news_id": hashlib.sha256(f"{source['source']}::{url}".encode()).hexdigest(), "source": source["source"],
        "url": url, "title": title, "summary": summary, "content": content,
        "published_date": published.isoformat() if published else "", "published_at_raw": published_at_raw,
        "author": _extract_author(doc), "category": _extract_category(doc),
        "crawl_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "crawl_status": "ok",
        "error_message": "", "content_hash": hashlib.sha256(" ".join(article_text.split()).encode()).hexdigest(),
    }


def _candidate_links(seed_url: str, domain: str, doc) -> list[str]:
    ranked: list[tuple[int, str]] = []
    seed_parts = [part for part in urlsplit(seed_url).path.lower().split("/") if part]
    topic = seed_parts[0].removesuffix(".htm") if seed_parts else ""
    for anchor in doc.xpath('//a[@href]'):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        url = _normalize_url(urljoin(seed_url, href))
        path = urlsplit(url).path.lower()
        if not _same_domain(url, domain) or path.endswith(SKIP_EXTENSIONS):
            continue
        anchor_text = " ".join(anchor.itertext()).strip()
        score = (1 if path not in {"", "/"} else 0) + (2 if len(anchor_text) >= 20 else 0) + (1 if re.search(r"\d{4}|\d{6,}|\.htm", path) else 0) + (1 if path.count("-") >= 3 else 0) + (3 if topic and topic in path else 0)
        if score >= 1 and url != _normalize_url(seed_url):
            ranked.append((score, url))
    seen: set[str] = set()
    return [url for _, url in sorted(ranked, key=lambda item: item[0], reverse=True) if not (url in seen or seen.add(url))]


def _valid_article(row: dict, start_date: date | None, end_date: date | None) -> tuple[bool, str]:
    if len(row["title"]) < 8:
        return False, "missing/short title"
    if len(row["content"]) < 120:
        return False, "missing/short article body"
    if not row["published_date"]:
        return False, "missing/unparseable publication date"
    published = date.fromisoformat(row["published_date"])
    if start_date and published < start_date or end_date and published > end_date:
        return False, "outside requested date range"
    return True, ""


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def crawl_sources(sources: Iterable[dict] | None = None, max_articles_per_source: int = 10, output_path: Path | str | None = None, log_path: Path | str | None = None, report_path: Path | str | None = None, start_date: date | None = None, end_date: date | None = None, retries: int = 3, delay: float = 0.5) -> list[dict]:
    output_path, log_path, report_path = Path(output_path or BACKFILL_PATH), Path(log_path or LOG_PATH), Path(report_path or REPORT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stock_news_crawler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    rows: list[dict] = []
    reports: list[dict] = []
    seen_urls: set[str] = set()
    for source in list(sources) if sources is not None else _read_sources():
        name, seed, domain = source.get("source", "unknown"), source.get("seed_url", "").strip(), source.get("domain", "").strip()
        attempted = rejected = 0
        errors: list[str] = []
        try:
            if not seed or not domain:
                raise ValueError("missing seed_url or domain")
            seed_doc = html.fromstring(_safe_fetch(seed))
            candidates = _candidate_links(seed, domain, seed_doc)
            saved = 0
            for article_url in candidates:
                if saved >= max_articles_per_source or attempted >= max(10, max_articles_per_source * 5):
                    break
                if article_url in seen_urls:
                    continue
                attempted += 1
                try:
                    row = _standardize_row(source, article_url, html.fromstring(_safe_fetch(article_url)))
                    valid, reason = _valid_article(row, start_date, end_date)
                    if not valid:
                        rejected += 1
                        logger.warning("Rejected %s (%s): %s", article_url, name, reason)
                        continue
                    if row["url"] in seen_urls:
                        continue
                    rows.append(row)
                    seen_urls.add(row["url"])
                    saved += 1
                    logger.info("Crawled %s: %s", name, row["url"])
                    if delay:
                        time.sleep(delay)
                except Exception as exc:
                    rejected += 1
                    errors.append(f"{article_url}: {exc}")
                    logger.error("Failed article %s (%s): %s", article_url, name, exc)
            status = "ok" if saved else "no_valid_articles"
            if not candidates:
                errors.append("no candidate article links found on seed page")
        except Exception as exc:
            saved, status = 0, "seed_error"
            errors.append(str(exc))
            logger.error("Failed seed %s (%s): %s", seed, name, exc)
        reports.append({"source": name, "seed_url": seed, "status": status, "articles_saved": saved, "pages_attempted": attempted, "pages_rejected": rejected, "error_message": " | ".join(errors[:5])})
    _write_csv(output_path, rows, FIELDNAMES)
    _write_csv(report_path, reports, ["source", "seed_url", "status", "articles_saved", "pages_attempted", "pages_rejected", "error_message"])
    logger.info("Saved %s verified rows to %s", len(rows), output_path)
    return rows


def main() -> None:
    config = _load_config()
    parser = argparse.ArgumentParser(description="Crawl verified news articles from configured sources")
    parser.add_argument("--max-per-source", type=int, default=int(config.get("max_articles_per_source", 10)))
    parser.add_argument("--start-date", default=config.get("start_date"))
    parser.add_argument("--end-date", default=config.get("end_date", "today"))
    parser.add_argument("--output", type=Path, default=BACKFILL_PATH)
    args = parser.parse_args()
    if args.output.resolve() == RAW_DATA_PATH.resolve():
        parser.error(f"refusing to overwrite {RAW_DATA_PATH}; use scrape_native_candidates.py to append verified articles")
    start = date.fromisoformat(args.start_date) if args.start_date else None
    end = date.today() if args.end_date == "today" else date.fromisoformat(args.end_date)
    crawl_sources(max_articles_per_source=args.max_per_source, output_path=args.output, start_date=start, end_date=end, retries=int(config.get("max_retries", 3)), delay=float(config.get("request_delay_seconds", 0.5)))


if __name__ == "__main__":
    main()
