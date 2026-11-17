"""Focused integration checks for the candidate-free review38 finalizer."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.annotation_utils import NEWS_COLUMNS
from src.finalize_news_type_review import (
    EXPECTED_COUNTS,
    EXPECTED_RESOLUTION_SHA256,
    EXPECTED_WORKBOOK_SHA256,
    RESOLUTION_COLUMNS,
    finalize_news_type_review,
)
from src.cleaning_utils import APPROVED_DUPLICATE_REASON
from src.news_type_inference import FINAL_OUTPUT_COLUMNS, OUTPUT_COLUMNS


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATHS = [
    "data/annotation/news_type/review38/review38_source.xlsx",
    "data/annotation/news_type/review38/review38_resolution.csv",
    "data/annotation/news_type/news_manually_labeled.csv",
    "data/processed/news_clean.csv",
    "data/interim/news_excluded_records.csv",
    "data/interim/duplicate_merge_log.csv",
    "models/news_type_phobert_cnn/label_mapping.json",
    "data/processed/news_type_predictions.csv",
    "data/processed/news_type_predictions_metadata.json",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_project_inputs(tmp_path: Path) -> Path:
    project = tmp_path / "news_processing"
    for relative in INPUT_PATHS:
        source = PROJECT_DIR / relative
        assert source.is_file(), f"Required finalizer fixture is missing: {source}"
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return project


def test_review38_finalizer_publishes_manual_model_and_explicit_excluded_rows(tmp_path):
    project = _copy_project_inputs(tmp_path)
    output_path = project / "data/processed/news_type_predictions.csv"
    metadata_path = project / "data/processed/news_type_predictions_metadata.json"
    master_path = project / "data/annotation/news_type/news_manually_labeled.csv"
    clean_path = project / "data/processed/news_clean.csv"
    resolution_path = project / "data/annotation/news_type/review38/review38_resolution.csv"
    resolution_hash_before = _sha256(resolution_path)
    assert resolution_hash_before == EXPECTED_RESOLUTION_SHA256

    before = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    before_model = before.loc[
        before["label_source"].eq("model"),
        ["news_id", "news_type", "prediction_score"],
    ].reset_index(drop=True)
    result = finalize_news_type_review(project)

    output = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    master = pd.read_csv(master_path, dtype=str, keep_default_na=False)
    clean = pd.read_csv(clean_path, dtype=str, keep_default_na=False)
    resolution = pd.read_csv(resolution_path, dtype=str, keep_default_na=False)

    assert result["counts"] == EXPECTED_COUNTS
    assert len(output) == 3354
    assert output.columns.tolist() == FINAL_OUTPUT_COLUMNS
    assert output["news_id"].is_unique
    pd.testing.assert_frame_equal(output[NEWS_COLUMNS], clean[NEWS_COLUMNS])
    assert output["label_source"].value_counts().to_dict() == {
        "model": 2260,
        "manual": 1090,
        "excluded": 4,
    }
    after_model = output.loc[
        output["label_source"].eq("model"),
        ["news_id", "news_type", "prediction_score"],
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(after_model, before_model)

    assert resolution.columns.tolist() == RESOLUTION_COLUMNS
    assert resolution["final_annotation_status"].value_counts().to_dict() == {
        "labeled": 34,
        "excluded": 4,
    }
    labeled = resolution["final_annotation_status"].eq("labeled")
    excluded = resolution["final_annotation_status"].eq("excluded")
    assert resolution.loc[labeled, "final_news_type"].ne("").all()
    assert resolution.loc[excluded, "final_news_type"].eq("").all()
    assert resolution.loc[excluded, "exclusion_reason"].ne("").all()

    output_by_id = output.set_index("news_id")
    labeled_ids = resolution.loc[labeled, "news_id"].tolist()
    duplicate_alias_ids = resolution.loc[
        resolution["v2_action"].eq("DROP_DUPLICATE"), "news_id"
    ].tolist()
    canonical_labeled_ids = [
        news_id for news_id in labeled_ids if news_id not in duplicate_alias_ids
    ]
    excluded_ids = resolution.loc[excluded, "news_id"].tolist()
    assert not set(duplicate_alias_ids).intersection(output_by_id.index)
    assert output_by_id.loc[canonical_labeled_ids, "label_source"].eq("manual").all()
    assert (
        output_by_id.loc[canonical_labeled_ids, "news_type"].tolist()
        == resolution.set_index("news_id").loc[
            canonical_labeled_ids, "final_news_type"
        ].tolist()
    )
    assert output_by_id.loc[excluded_ids, "label_source"].eq("excluded").all()
    assert output_by_id.loc[excluded_ids, ["news_type", "prediction_score"]].eq("").all().all()
    assert output.loc[output["label_source"].eq("manual"), "prediction_score"].eq("").all()
    duplicate_decisions = resolution.loc[
        resolution["v2_action"].eq("DROP_DUPLICATE"),
        ["matched_news_id", "final_news_type"],
    ]
    representative_ids = duplicate_decisions["matched_news_id"].tolist()
    assert output_by_id.loc[representative_ids, "label_source"].eq("manual").all()
    assert (
        output_by_id.loc[representative_ids, "news_type"].tolist()
        == duplicate_decisions["final_news_type"].tolist()
    )

    master_by_id = master.set_index("news_id")
    assert master_by_id.loc[labeled_ids, "annotation_status"].eq("labeled").all()
    assert master_by_id.loc[excluded_ids, "annotation_status"].eq("excluded").all()
    assert metadata["schema_version"] == "news_type_predictions_v3"
    assert metadata["source_run"] == "20261024_153932_380173"
    assert metadata["selected_model_unchanged"] is True
    assert metadata["label_source_counts"] == EXPECTED_COUNTS
    assert metadata["output_columns"] == FINAL_OUTPUT_COLUMNS
    assert metadata["output_csv_sha256"] == _sha256(output_path)
    assert metadata["review_batch"]["source_workbook_sha256"] == EXPECTED_WORKBOOK_SHA256
    assert metadata["review_batch"]["manual_labels"] == 34
    assert metadata["review_batch"]["canonical_review_articles"] == 32
    assert metadata["review_batch"]["duplicate_aliases"] == 2
    assert metadata["review_batch"]["resolution_sha256"] == _sha256(resolution_path)
    assert _sha256(resolution_path) == resolution_hash_before
    assert set(metadata["excluded_news_ids"]) == set(excluded_ids)
    serialized_metadata = json.dumps(metadata, ensure_ascii=False)
    for forbidden in (
        "candidate_model_run",
        "candidate_accepted",
        "candidate_test_evaluated",
        "added_train_rows",
        "manual_only_rows",
        "selection_report",
    ):
        assert forbidden not in serialized_metadata

    tracked = [resolution_path, master_path, output_path, metadata_path]
    first_hashes = {path: _sha256(path) for path in tracked}
    second_result = finalize_news_type_review(project)
    assert second_result == result
    assert {path: _sha256(path) for path in tracked} == first_hashes
    assert _sha256(resolution_path) == resolution_hash_before


def test_review38_finalizer_migrates_compact_v2_output_to_joined_v3(tmp_path):
    project = _copy_project_inputs(tmp_path)
    output_path = project / "data/processed/news_type_predictions.csv"
    metadata_path = project / "data/processed/news_type_predictions_metadata.json"
    output = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    compact = output[OUTPUT_COLUMNS]
    compact.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    metadata["schema_version"] = "news_type_predictions_v2"
    metadata["output_columns"] = OUTPUT_COLUMNS
    metadata["output_csv_sha256"] = _sha256(output_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    finalize_news_type_review(project)

    migrated = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    migrated_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert migrated.columns.tolist() == FINAL_OUTPUT_COLUMNS
    assert migrated_metadata["schema_version"] == "news_type_predictions_v3"
    assert migrated_metadata["output_columns"] == FINAL_OUTPUT_COLUMNS
    assert migrated_metadata["output_csv_sha256"] == _sha256(output_path)


def test_review38_finalizer_rejects_a_tampered_source_before_writing(tmp_path):
    project = _copy_project_inputs(tmp_path)
    workbook_path = project / "data/annotation/news_type/review38/review38_source.xlsx"
    master_path = project / "data/annotation/news_type/news_manually_labeled.csv"
    output_path = project / "data/processed/news_type_predictions.csv"
    metadata_path = project / "data/processed/news_type_predictions_metadata.json"
    resolution_path = project / "data/annotation/news_type/review38/review38_resolution.csv"
    before_hashes = {
        path: _sha256(path)
        for path in (resolution_path, master_path, output_path, metadata_path)
    }
    with workbook_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        finalize_news_type_review(project)

    assert {
        path: _sha256(path)
        for path in (resolution_path, master_path, output_path, metadata_path)
    } == before_hashes


@pytest.mark.parametrize("tamper", ["missing_column", "wrong_action"])
def test_review38_finalizer_rejects_tampered_v2_disposition_before_writing(
    tmp_path, tamper,
):
    project = _copy_project_inputs(tmp_path)
    resolution_path = project / "data/annotation/news_type/review38/review38_resolution.csv"
    output_path = project / "data/processed/news_type_predictions.csv"
    metadata_path = project / "data/processed/news_type_predictions_metadata.json"
    resolution = pd.read_csv(resolution_path, dtype=str, keep_default_na=False)
    if tamper == "missing_column":
        resolution = resolution.drop(columns="matched_news_id")
    else:
        resolution.loc[resolution["v2_action"].eq("TRAIN_V2"), "v2_action"] = (
            "DROP_DUPLICATE"
        )
    resolution.to_csv(resolution_path, index=False, encoding="utf-8-sig")
    before_hashes = {
        path: _sha256(path) for path in (output_path, metadata_path)
    }

    with pytest.raises(ValueError, match=(
        "16-column|four V2" if tamper == "missing_column" else "disposition counts"
    )):
        finalize_news_type_review(
            project,
            expected_resolution_sha256=_sha256(resolution_path),
        )

    assert {
        path: _sha256(path) for path in (output_path, metadata_path)
    } == before_hashes


def test_review38_finalizer_rejects_semantically_valid_resolution_change_by_hash(
    tmp_path,
):
    project = _copy_project_inputs(tmp_path)
    resolution_path = project / "data/annotation/news_type/review38/review38_resolution.csv"
    output_path = project / "data/processed/news_type_predictions.csv"
    metadata_path = project / "data/processed/news_type_predictions_metadata.json"
    resolution = pd.read_csv(resolution_path, dtype=str, keep_default_na=False)
    heldout_index = resolution.index[resolution["v2_action"].eq("HELDOUT_OVERLAP")][0]
    resolution.loc[heldout_index, "matched_news_id"] = "semantically-valid-but-unlocked-id"
    resolution.to_csv(resolution_path, index=False, encoding="utf-8-sig")
    before_hashes = {
        path: _sha256(path) for path in (output_path, metadata_path)
    }

    with pytest.raises(ValueError, match="resolution SHA-256 mismatch"):
        finalize_news_type_review(project)

    assert {
        path: _sha256(path) for path in (output_path, metadata_path)
    } == before_hashes


@pytest.mark.parametrize("tamper", ["missing_group", "extra_member", "wrong_count"])
def test_review38_finalizer_rejects_invalid_duplicate_lineage_before_writing(
    tmp_path, tamper,
):
    project = _copy_project_inputs(tmp_path)
    duplicate_path = project / "data/interim/duplicate_merge_log.csv"
    output_path = project / "data/processed/news_type_predictions.csv"
    metadata_path = project / "data/processed/news_type_predictions_metadata.json"
    duplicate_log = pd.read_csv(duplicate_path, dtype=str, keep_default_na=False)
    approved = duplicate_log["duplicate_reasons"].eq(APPROVED_DUPLICATE_REASON)
    target_index = duplicate_log.index[approved][0]
    if tamper == "missing_group":
        duplicate_log = duplicate_log.drop(index=target_index)
    elif tamper == "extra_member":
        duplicate_log.loc[target_index, "merged_news_ids"] += "|unexpected-id"
    else:
        duplicate_log.loc[target_index, "rows_removed"] = "2"
    duplicate_log.to_csv(duplicate_path, index=False, encoding="utf-8-sig")
    before_hashes = {
        path: _sha256(path) for path in (output_path, metadata_path)
    }

    with pytest.raises(ValueError, match="merge lineage|lineage disagrees"):
        finalize_news_type_review(project)

    assert {
        path: _sha256(path) for path in (output_path, metadata_path)
    } == before_hashes
