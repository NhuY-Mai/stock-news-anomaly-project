"""Notebook-level V2 metadata must agree with the locked test manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_05 = PROJECT_DIR / "notebooks/phase_1_news_type_v2/05_train_phobert_cnn_v2.ipynb"
NOTEBOOK_06 = PROJECT_DIR / "notebooks/phase_1_news_type_v2/06_evaluate_phobert_cnn_v2.ipynb"
REPORT_DIR = PROJECT_DIR / "reports/news_type/v2/runs/review38_incremental_v2"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@pytest.mark.skipif(
    not (NOTEBOOK_06.is_file() and (REPORT_DIR / "test_evaluation_config.json").is_file()),
    reason="Thiếu notebook hoặc manifest test V2.",
)
def test_notebook_06_metadata_matches_test_manifest():
    metadata = _json(NOTEBOOK_06)["metadata"]["v2_metadata"]
    manifest = _json(REPORT_DIR / "test_evaluation_config.json")

    assert metadata["run_name"] == manifest["source_run"]
    assert metadata["parent_run"] == manifest["parent_run"]
    assert metadata["test_evaluated"] is manifest["test_evaluated"] is True
    assert metadata["test_evaluated_at_utc"] == manifest["created_at_utc"]
    assert metadata["test_used_for_model_selection"] is manifest["test_used_for_model_selection"] is False
    assert (PROJECT_DIR / metadata["test_evaluation_manifest"]).resolve() == (
        REPORT_DIR / "test_evaluation_config.json"
    ).resolve()


@pytest.mark.skipif(not NOTEBOOK_05.is_file(), reason="Thiếu notebook 05 V2.")
def test_training_stage_records_stay_pre_test():
    """Train/selection records describe the stage before the post-hoc test."""
    for path in (
        PROJECT_DIR / "models/news_type_phobert_cnn/v2/runs/review38_incremental_v2/run_config.json",
        REPORT_DIR / "selection_decision.json",
    ):
        if path.is_file():
            assert _json(path).get("test_evaluated") is False, path
