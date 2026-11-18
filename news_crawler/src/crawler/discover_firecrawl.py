"""Discover auditable ticker-news URLs on the 15 configured domains with Firecrawl."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from .run_backfill import ROOT, SOURCES_PATH, _normalize_url, _same_domain

ALIASES_PATH = ROOT / "config" / "ticker_aliases.json"
FIRECRAWL_DIR = ROOT / ".firecrawl"
CANDIDATES_PATH = ROOT / "data" / "staging" / "news_candidates.csv"
SKIP_PATHS = ("/du-lieu/", "/gia-co-phieu", "/lich-su-giao-dich", "/co-dong", "/bao-cao-tai-chinh", "/profile/")
MARKET_CONTEXT = ("co phieu", "ma chung khoan", "hose", "hnx", "upcom", "chung khoan")


def _firecrawl_command(*arguments: str) -> list[str]:
    """Bypass the PowerShell npm shim so subprocess preserves spaced queries."""
    cmd_shim = shutil.which("firecrawl.cmd") or shutil.which("firecrawl")
    node = shutil.which("node.exe") or shutil.which("node")
    if not cmd_shim or not node:
        raise RuntimeError("firecrawl or node.exe was not found on PATH")
    entrypoint = Path(cmd_shim).parent / "node_modules" / "firecrawl-cli" / "dist" / "index.js"
    if not entrypoint.exists():
        raise RuntimeError(f"Firecrawl CLI entrypoint not found: {entrypoint}")
    return [node, str(entrypoint), *arguments]


def _fold(value: str) -> str:
    value = unicodedata.normalize("NFD", value.casefold())
    return " ".join("".join(char for char in value if unicodedata.category(char) != "Mn").split())


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sources(path: Path = SOURCES_PATH) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("enabled", "").lower() == "true"]


def match_tickers(title: str, description: str, url: str, aliases: dict[str, list[str]]) -> list[dict]:
    text = _fold(f"{title} {description}")
    path = _fold(urlsplit(url).path.replace("-", " ").replace("_", " "))
    matches: list[dict] = []
    for ticker, names in aliases.items():
        for alias in names:
            folded = _fold(alias)
            if re.search(rf"(?<!\w){re.escape(folded)}(?!\w)", text):
                matches.append({"ticker": ticker, "matched_text": alias, "match_method": "company_alias", "match_confidence": "high"})
                break
        else:
            ticker_pattern = rf"(?<![a-z0-9]){ticker.casefold()}(?![a-z0-9])"
            if re.search(ticker_pattern, text) and any(context in text for context in MARKET_CONTEXT):
                matches.append({"ticker": ticker, "matched_text": ticker, "match_method": "ticker_with_market_context", "match_confidence": "high"})
            elif re.search(ticker_pattern, path) and any(context in text for context in MARKET_CONTEXT):
                matches.append({"ticker": ticker, "matched_text": ticker, "match_method": "ticker_in_url_with_market_context", "match_confidence": "medium"})
    return matches


def is_article_candidate(url: str, domain: str) -> bool:
    path = urlsplit(url).path.casefold()
    return _same_domain(url, domain) and path not in {"", "/"} and not any(marker in path for marker in SKIP_PATHS)


def _search_query(domain: str, aliases: dict[str, list[str]], start: str, end: str) -> str:
    # Tickers keep the query short; company aliases are applied locally to search metadata.
    terms = " OR ".join(aliases)
    return f"site:{domain} ({terms}) (cổ phiếu OR doanh nghiệp OR lợi nhuận OR doanh thu) after:{start} before:{end}"


def _run_search(source: dict, aliases: dict[str, list[str]], start: str, end: str, limit: int) -> tuple[dict, Path]:
    FIRECRAWL_DIR.mkdir(parents=True, exist_ok=True)
    output = FIRECRAWL_DIR / f"search-{source['source'].lower()}.json"
    command = _firecrawl_command("search", _search_query(source["domain"], aliases, start, end), "--limit", str(limit), "--sources", "web", "--country", "VN", "--no-highlights", "-o", str(output), "--json")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    if not output.exists():
        return {"data": {"web": []}}, output
    return json.loads(output.read_text(encoding="utf-8")), output


def _send_feedback(payload: dict, matched_urls: list[str]) -> None:
    search_id = payload.get("id")
    if not search_id:
        return
    if matched_urls:
        command = _firecrawl_command("search-feedback", search_id, "--rating", "good", "--valuable-sources", json.dumps([{"url": matched_urls[0], "reason": "Ticker-linked news candidate on the requested domain"}], ensure_ascii=False), "--silent")
    else:
        command = _firecrawl_command("search-feedback", search_id, "--rating", "bad", "--missing-content", json.dumps([{"topic": "ticker-linked articles", "description": "No auditable article matching the requested stock universe"}], ensure_ascii=False), "--silent")
    subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)


def discover(start: str, end: str, limit: int = 100, max_per_ticker_source: int = 0, source_names: set[str] | None = None, output_path: Path = CANDIDATES_PATH) -> list[dict]:
    aliases, sources = load_aliases(), load_sources()
    if source_names:
        sources = [source for source in sources if source["source"] in source_names]
    rows: list[dict] = []
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for source in sources:
        try:
            payload, _ = _run_search(source, aliases, start, end, limit)
            results = payload.get("data", {}).get("web", [])
        except Exception as exc:
            rows.append({"source": source["source"], "ticker": "", "url": "", "title": "", "description": "", "matched_text": "", "match_method": "search_error", "match_confidence": "", "search_error": str(exc)})
            continue
        matched_urls: list[str] = []
        for result in results:
            url = _normalize_url(result.get("url", ""))
            if not url or not is_article_candidate(url, source["domain"]):
                continue
            for match in match_tickers(result.get("title", ""), result.get("description", ""), url, aliases):
                key = (source["source"], match["ticker"])
                if max_per_ticker_source > 0 and counts[key] >= max_per_ticker_source:
                    continue
                counts[key] += 1
                rows.append({"source": source["source"], "ticker": match["ticker"], "url": url, "title": result.get("title", ""), "description": result.get("description", ""), **match, "match_confidence": match["match_confidence"], "search_error": ""})
                matched_urls.append(url)
        _send_feedback(payload, matched_urls)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "ticker", "url", "title", "description", "matched_text", "match_method", "match_confidence", "search_error"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-09-26")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-per-ticker-source", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--sources", help="Comma-separated configured source names")
    parser.add_argument("--output", type=Path, default=CANDIDATES_PATH)
    args = parser.parse_args()
    selected = set(args.sources.split(",")) if args.sources else None
    rows = discover(args.start_date, args.end_date, args.limit, args.max_per_ticker_source, selected, args.output)
    valid = [row for row in rows if row["ticker"]]
    print(json.dumps({"candidates": len(valid), "sources": len({row['source'] for row in valid}), "tickers": len({row['ticker'] for row in valid})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
