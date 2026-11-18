"""Resume the unlimited, Python-only news backfill as a durable pipeline."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG = ROOT / "outputs" / "logs" / "native_pipeline.log"
STATUS = ROOT / "outputs" / "logs" / "native_pipeline_status.json"


def record(stage: str, state: str, returncode: int | None = None) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({
        "stage": stage,
        "state": state,
        "returncode": returncode,
        "updated_at": datetime.now().astimezone().isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def run(stage: str, args: list[str], log) -> None:
    record(stage, "running")
    log.write(f"\n[{datetime.now().astimezone().isoformat()}] START {stage}\n")
    log.flush()
    result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        record(stage, "failed", result.returncode)
        raise SystemExit(result.returncode)
    record(stage, "complete", 0)


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        run("discover", [sys.executable, "-m", "src.crawler.discover_native"], log)
        run("scrape", [sys.executable, "-m", "src.crawler.scrape_native_candidates", "--candidates", "data/staging/news_candidates_native.csv", "--workers", "12"], log)
        run("coverage", [sys.executable, "-m", "src.quality.build_coverage"], log)
        run("tests", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp=.test_tmp_native_all"], log)
    record("pipeline", "complete", 0)


if __name__ == "__main__":
    main()
