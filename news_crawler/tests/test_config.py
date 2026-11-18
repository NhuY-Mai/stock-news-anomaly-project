import json
from pathlib import Path

from src.crawler.run_backfill import crawl_sources
from src.crawler.run_backfill import _normalize_url, _same_domain
from src.crawler.discover_firecrawl import is_article_candidate, match_tickers


def test_project_config_has_required_tickers():
    path = Path(__file__).parents[1] / "config" / "project_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert len(config["tickers"]) == 14
    assert config["start_date"] == "2020-01-01"


def test_crawl_sources_writes_csv(tmp_path, monkeypatch):
    source = {
        "source": "ExampleNews",
        "domain": "example.com",
        "seed_url": "https://example.com/",
        "enabled": "true",
        "source_type": "category",
    }

    fake_pages = {
        "https://example.com/": '''
            <html>
              <body>
                <a href="/story-1">Story 1</a>
              </body>
            </html>
        ''',
        "https://example.com/story-1": '''
            <html>
              <head>
                <meta property="og:title" content="AAPL jumps on strong earnings" />
                <meta name="description" content="A strong earnings story for AAPL." />
                <meta property="article:published_time" content="2024-01-02T10:00:00+00:00" />
              </head>
              <body>
                <article>
                  <p>Apple reported stronger earnings.</p>
                  <p>The company exceeded analysts' expectations.</p>
                  <p>Revenue and profit both increased, while management also raised its outlook for the next quarter.</p>
                </article>
              </body>
            </html>
        ''',
    }

    monkeypatch.setattr("src.crawler.run_backfill._safe_fetch", lambda url, timeout=20: fake_pages[url])

    output_path = tmp_path / "news_raw.csv"
    rows = crawl_sources(
        sources=[source],
        max_articles_per_source=1,
        output_path=output_path,
        log_path=tmp_path / "crawl.log",
    )

    assert len(rows) == 1
    assert rows[0]["source"] == "ExampleNews"
    assert rows[0]["title"] == "AAPL jumps on strong earnings"
    assert output_path.exists()


def test_tracking_parameters_are_removed_without_changing_real_query():
    url = _normalize_url("https://cafef.vn/a.chn?id=7&utm_source=test#fragment")
    assert url == "https://cafef.vn/a.chn?id=7"


def test_domain_check_rejects_lookalike_domain():
    assert _same_domain("https://m.cafef.vn/a.chn", "cafef.vn")
    assert not _same_domain("https://cafef.vn.evil.example/a", "cafef.vn")


def test_ticker_matching_requires_company_alias_or_market_context():
    aliases = {"MWG": ["Thế Giới Di Động"], "IMP": ["Imexpharm"]}
    matched = match_tickers("MWG tăng trần", "Cổ phiếu bán lẻ tăng mạnh", "https://example.com/mwg-tang-tran", aliases)
    assert [item["ticker"] for item in matched] == ["MWG"]
    assert match_tickers("An important update", "General world news", "https://example.com/important", aliases) == []


def test_stock_profile_is_not_an_article_candidate():
    assert not is_article_candidate("https://cafef.vn/du-lieu/hose/mwg.chn", "cafef.vn")
    assert is_article_candidate("https://cafef.vn/mwg-bao-lai-lon-188.chn", "cafef.vn")
