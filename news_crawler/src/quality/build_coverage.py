"""Build an explicit configured-source x ticker coverage matrix."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    with (ROOT / "config" / "sources.csv").open(encoding="utf-8-sig", newline="") as handle:
        sources = [row["source"] for row in csv.DictReader(handle) if row.get("enabled", "").lower() == "true"]
    tickers = json.loads((ROOT / "config" / "project_config.json").read_text(encoding="utf-8"))["tickers"]
    with (ROOT / "data" / "final" / "news_ticker.csv").open(encoding="utf-8-sig", newline="") as handle:
        links = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in links:
        grouped[(row["source"], row["ticker"])].append(row)
    rows = []
    for source in sources:
        for ticker in tickers:
            items = grouped[(source, ticker)]
            dates = [item["published_date"] for item in items if item["published_date"]]
            rows.append({"source": source, "ticker": ticker, "first_date": min(dates) if dates else "", "last_date": max(dates) if dates else "", "article_count": len({item["news_id"] for item in items}), "unique_url_count": len({item["url"] for item in items}), "content_count": len({item["news_id"] for item in items}), "error_count": 0, "coverage_status": "covered" if items else "missing"})
    output = ROOT / "outputs" / "coverage" / "coverage.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} source-ticker coverage rows to {output}")


if __name__ == "__main__":
    main()
