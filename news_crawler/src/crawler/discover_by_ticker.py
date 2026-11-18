"""Use one broad Firecrawl search per ticker, then retain configured publishers."""
from __future__ import annotations
import argparse, csv, json, subprocess
from pathlib import Path
from urllib.parse import urlsplit
from .discover_firecrawl import _firecrawl_command, _send_feedback, is_article_candidate, load_aliases, load_sources, match_tickers
from .run_backfill import ROOT, _normalize_url

OUTPUT = ROOT / "data" / "staging" / "news_candidates_by_ticker.csv"


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--limit',type=int,default=100); args=parser.parse_args()
    aliases=load_aliases(); sources=load_sources(); domain_map={s['domain']:s for s in sources}; rows=[]
    outdir=ROOT/'.firecrawl'/'ticker-search'; outdir.mkdir(parents=True,exist_ok=True)
    for ticker,names in aliases.items():
        query=f'{ticker} {names[0]} cổ phiếu doanh thu lợi nhuận after:2020-01-01 before:2026-09-26'
        output=outdir/f'{ticker.lower()}.json'
        cmd=_firecrawl_command('search',query,'--limit',str(args.limit),'--sources','web','--country','VN','--no-highlights','-o',str(output),'--json')
        run=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=120)
        if run.returncode or not output.exists(): continue
        payload=json.loads(output.read_text(encoding='utf-8')); matched_urls=[]
        for item in payload.get('data',{}).get('web',[]):
            url=_normalize_url(item.get('url','')); host=(urlsplit(url).hostname or '').removeprefix('www.')
            source=next((s for d,s in domain_map.items() if host==d or host.endswith('.'+d)),None)
            if not source or not is_article_candidate(url,source['domain']): continue
            matches=match_tickers(item.get('title',''),item.get('description',''),url,{ticker:names})
            if not matches: continue
            m=matches[0]; rows.append({'source':source['source'],'ticker':ticker,'url':url,'title':item.get('title',''),'description':item.get('description',''),'matched_text':m['matched_text'],'match_method':m['match_method'],'match_confidence':m['match_confidence'],'search_error':''}); matched_urls.append(url)
        _send_feedback(payload,matched_urls)
    unique={(r['source'],r['ticker'],r['url']):r for r in rows}; OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    fields=['source','ticker','url','title','description','matched_text','match_method','match_confidence','search_error']
    with OUTPUT.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(unique.values())
    print(json.dumps({'candidates':len(unique),'sources':len({r['source'] for r in unique.values()}),'tickers':len({r['ticker'] for r in unique.values()})},ensure_ascii=False))

if __name__=='__main__': main()
