"""Discover stock-news URLs with Python only (sitemaps, RSS and seed pages)."""
from __future__ import annotations
import csv, gzip, json, re, sqlite3
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from lxml import etree, html

from .discover_firecrawl import _fold, is_article_candidate, load_aliases, load_sources, match_tickers
from .run_backfill import RAW_DATA_PATH, ROOT, _normalize_url, _safe_fetch

OUTPUT = ROOT / "data" / "staging" / "news_candidates_native.csv"
STATE = ROOT / "data" / "staging" / "native_discovery.sqlite"
FEED_PATHS = ("/rss", "/rss.xml", "/feed", "/feed.xml", "/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml")


def _db():
    conn=sqlite3.connect(STATE)
    conn.execute("create table if not exists visited(url text primary key, status text, checked_at text default current_timestamp)")
    conn.execute("""create table if not exists candidates(
        source text, ticker text, url text, title text, description text,
        matched_text text, match_method text, match_confidence text,
        search_error text, discovered_at text default current_timestamp,
        primary key(source,ticker,url))""")
    return conn


def _outside_project_years(text):
    """Reject a dated sitemap/entry only when its year is unambiguous."""
    years=[int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", text or "")]
    return bool(years) and all(y < 2020 or y > 2026 for y in years)


def _save_candidate(conn,row):
    fields=("source","ticker","url","title","description","matched_text","match_method","match_confidence","search_error")
    conn.execute(
        f"insert or ignore into candidates({','.join(fields)}) values({','.join('?' for _ in fields)})",
        tuple(row[k] for k in fields),
    )


def _known_urls():
    if not RAW_DATA_PATH.exists(): return set()
    with RAW_DATA_PATH.open(encoding="utf-8-sig",newline="") as f:return {_normalize_url(r["url"]) for r in csv.DictReader(f)}


def _fetch(url):
    try:return _safe_fetch(url,timeout=15,retries=2,delay=1),""
    except Exception as exc:return "",str(exc)


def _relevant(text,url,aliases):
    matches=match_tickers(text,"",url,aliases)
    if matches:return matches
    folded=_fold(urlsplit(url).path.replace('-',' ').replace('_',' '))
    out=[]
    for ticker,names in aliases.items():
        tokens=[ticker.casefold(),*[_fold(n) for n in names]]
        if any(re.search(rf"(?<!\w){re.escape(token)}(?!\w)",folded) for token in tokens):
            out.append({"ticker":ticker,"matched_text":ticker,"match_method":"alias_in_url","match_confidence":"medium"})
    return out


def _extract(body,url,source,aliases,known):
    """Return child sitemaps and verified-by-metadata candidate rows."""
    children=[]; candidates=[]
    stripped=body.lstrip()
    if stripped.startswith("<"):
        try:
            root=etree.fromstring(body.encode("utf-8",errors="ignore"),parser=etree.XMLParser(recover=True,resolve_entities=False))
            local=etree.QName(root).localname.lower()
        except Exception: root=None; local=""
        if root is not None and local in {"sitemapindex","urlset","rss","feed"}:
            if local=="sitemapindex":
                for loc in root.xpath('//*[local-name()="sitemap"]/*[local-name()="loc"]/text()'):
                    child=str(loc).strip()
                    if not _outside_project_years(child): children.append(child)
                return children,candidates
            entries=root.xpath('//*[local-name()="url"] | //item | //*[(local-name()="entry")]')
            for entry in entries:
                loc=entry.xpath('./*[local-name()="loc" or local-name()="link"]/@href | ./*[local-name()="loc" or local-name()="link"]/text()')
                if not loc: continue
                article=_normalize_url(urljoin(url,str(loc[0]).strip()))
                title=" ".join(entry.xpath('.//*[local-name()="title"]/text()')).strip()
                lastmod=" ".join(entry.xpath('./*[local-name()="lastmod" or local-name()="pubDate" or local-name()="published"]/text()'))
                if _outside_project_years(lastmod) or article in known or not is_article_candidate(article,source['domain']): continue
                for m in _relevant(title,article,aliases):
                    candidates.append({"source":source['source'],"ticker":m['ticker'],"url":article,"title":title,"description":"","matched_text":m['matched_text'],"match_method":m['match_method'],"match_confidence":m['match_confidence'],"search_error":""})
            return children,candidates
    try: doc=html.fromstring(body)
    except Exception: return children,candidates
    for anchor in doc.xpath('//a[@href]'):
        article=_normalize_url(urljoin(url,anchor.get('href'))); title=" ".join(anchor.itertext()).strip()
        if article in known or not is_article_candidate(article,source['domain']): continue
        for m in _relevant(title,article,aliases):
            candidates.append({"source":source['source'],"ticker":m['ticker'],"url":article,"title":title,"description":"","matched_text":m['matched_text'],"match_method":m['match_method'],"match_confidence":m['match_confidence'],"search_error":""})
    return children,candidates


def discover(workers=32):
    aliases=load_aliases(); known=_known_urls(); conn=_db(); queue=deque(); queued=set()
    for source in load_sources():
        base=f"https://{source['domain']}"
        queue.append((source,base+"/robots.txt"))
        for path in FEED_PATHS: queue.append((source,base+path))
        queue.append((source,source["seed_url"]))
    with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
        while queue:
            batch=[]
            while queue and len(batch)<workers:
                source,url=queue.popleft(); url=_normalize_url(url)
                if url in queued: continue
                queued.add(url)
                if conn.execute("select 1 from visited where url=?",(url,)).fetchone(): continue
                batch.append((source,url))
            if not batch: continue
            results=list(pool.map(_fetch,[url for _,url in batch]))
            for (source,url),(body,error) in zip(batch,results):
                conn.execute("insert or replace into visited(url,status) values(?,?)",(url,"error" if error else "ok"))
                if not body: continue
                if urlsplit(url).path.rstrip('/').lower().endswith('robots.txt'):
                    for line in body.splitlines():
                        if line.lower().startswith("sitemap:"):
                            child=line.split(":",1)[1].strip()
                            if not _outside_project_years(child): queue.append((source,child))
                    continue
                children,candidates=_extract(body,url,source,aliases,known)
                queue.extend((source,child) for child in children)
                for row in candidates: _save_candidate(conn,row)
            conn.commit()
    fields=["source","ticker","url","title","description","matched_text","match_method","match_confidence","search_error"]
    rows=[dict(zip(fields,row)) for row in conn.execute(f"select {','.join(fields)} from candidates order by source,ticker,url")]
    with OUTPUT.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    conn.close(); print(json.dumps({"candidates":len(rows),"urls":len({r['url'] for r in rows}),"sources":len({r['source'] for r in rows})},ensure_ascii=False))


if __name__=='__main__': discover()
