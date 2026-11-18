"""Run unlimited-domain discovery, verification, and quality reporting."""
from __future__ import annotations
import json, subprocess, sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STATUS=ROOT/"outputs"/"logs"/"open_web_pipeline_status.json"
LOG=ROOT/"outputs"/"logs"/"open_web_pipeline.log"

def record(stage,state,code=None):
    STATUS.parent.mkdir(parents=True,exist_ok=True)
    STATUS.write_text(json.dumps({"stage":stage,"state":state,"returncode":code,"updated_at":datetime.now().astimezone().isoformat()},ensure_ascii=False,indent=2),encoding="utf-8")

def run(stage,args,log):
    record(stage,"running"); log.write(f"\nSTART {stage} {datetime.now().astimezone().isoformat()}\n"); log.flush()
    result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: record(stage,"failed",result.returncode); raise SystemExit(result.returncode)

def main():
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open("a",encoding="utf-8") as log:
        run("discover_open_web",[sys.executable,"-m","src.crawler.discover_open_web"],log)
        run("verify_articles",[sys.executable,"-m","src.crawler.scrape_native_candidates","--candidates","data/staging/news_candidates_open_web.csv","--workers","32"],log)
        run("coverage",[sys.executable,"-m","src.quality.build_coverage"],log)
    record("pipeline","complete",0)

if __name__=="__main__": main()
