"""Publish the reviewed News Type labels without changing the selected model.

This module applies one immutable 38-article human-review workbook, preserves
the already published baseline predictions byte-for-byte at row-value level,
and republishes the canonical assignment table.  It deliberately contains no
training, split-extension, candidate-model, or model-selection logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .annotation_utils import ANNOTATION_COLUMNS, NEWS_COLUMNS
from .news_type_inference import (
    FINAL_OUTPUT_COLUMNS,
    OUTPUT_COLUMNS,
    PREDICTION_COLUMNS,
    build_inference_plan,
    combine_news_type_predictions,
    join_news_type_output,
)


BASELINE_RUN = "20261024_153932_380173"
REVIEW_BATCH = "review38_v1"
REVIEW_APPLIED_DATE = "2026-11-05"
EXPECTED_WORKBOOK_SHA256 = (
    "898eff15947cf5f3fb55cb01116fe9161463bb6eff4f9f8b18fb4f1f1e094722"
)
EXPECTED_RESOLUTION_SHA256 = (
    "67329145acb296f65c1caa9be5bc6bf9c0f5da9f607d3fe22a68131b17e49616"
)
EXPECTED_COUNTS = {"manual": 1090, "model": 2260, "excluded": 4}
VALID_LABELS = {
    "governance_capital",
    "financial_results",
    "business_operations",
    "legal_compliance",
    "industry",
    "stock_market",
    "mention_only",
}
ARTICLE_COLUMNS = [
    "source",
    "url",
    "title",
    "summary",
    "content",
    "published_date",
    "tickers",
]
EXCLUSION_REASONS = {
    8: "ENTITY_LINK_UNVERIFIED",
    9: "MULTI_TOPIC_ARTICLE",
    23: "MULTI_TOPIC_ARTICLE",
    36: "MIXED_ARTICLE_FIELDS",
}
CORE_RESOLUTION_COLUMNS = [
    "review_batch",
    "applied_date",
    "stt",
    "news_id",
    "previous_news_type",
    "previous_annotation_status",
    "final_news_type",
    "final_annotation_status",
    "exclusion_reason",
    "annotator_note",
    "review_reason",
    "reviewed_content_sha256",
]
V2_DISPOSITION_COLUMNS = [
    "v2_action",
    "matched_split",
    "matched_news_id",
    "v2_reason",
]
RESOLUTION_COLUMNS = CORE_RESOLUTION_COLUMNS + V2_DISPOSITION_COLUMNS
EXPECTED_V2_ACTION_COUNTS = {
    "TRAIN_V2": 26,
    "HELDOUT_OVERLAP": 6,
    "DROP_DUPLICATE": 2,
    "EXCLUDED_SOURCE": 4,
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, lineterminator="\r\n")
    return b"\xef\xbb\xbf" + text.encode("utf-8")


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _atomic_write_if_changed(path: Path, payload: bytes) -> None:
    """Atomically replace one artifact, while making identical reruns a no-op."""
    if path.is_file() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_review(workbook_path: Path, expected_sha256: str) -> pd.DataFrame:
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Missing reviewed workbook: {workbook_path}")
    actual_hash = _sha256(workbook_path)
    if actual_hash != expected_sha256:
        raise ValueError(
            "Reviewed workbook SHA-256 mismatch: "
            f"expected {expected_sha256}, received {actual_hash}"
        )
    review = pd.read_excel(
        workbook_path, sheet_name="Gan_nhan", dtype=str, keep_default_na=False,
    )
    required = {
        "STT",
        "news_id",
        "news_type",
        "annotation_status",
        "annotator_note",
        "previous_news_type",
        "review_reason",
        *ARTICLE_COLUMNS,
    }
    missing = required.difference(review.columns)
    if missing:
        raise ValueError(f"Reviewed workbook is missing columns: {sorted(missing)}")
    review = review.copy()
    for column in required.difference({"STT"}):
        review[column] = review[column].map(_clean_text)
    review["STT"] = pd.to_numeric(review["STT"], errors="raise").astype(int)
    review = review.sort_values("STT", kind="stable").reset_index(drop=True)
    if review["STT"].tolist() != list(range(1, 39)):
        raise ValueError("Reviewed workbook must contain exactly STT 1..38")
    if not review["news_id"].is_unique or review["news_id"].eq("").any():
        raise ValueError("The 38 reviewed news_id values must be unique and nonempty")
    status_counts = review["annotation_status"].value_counts().to_dict()
    if status_counts != {"labeled": 34, "excluded": 4}:
        raise ValueError(f"Reviewed workbook must contain 34 labeled and 4 excluded: {status_counts}")
    excluded_stt = set(review.loc[review["annotation_status"].eq("excluded"), "STT"])
    if excluded_stt != set(EXCLUSION_REASONS):
        raise ValueError(f"Unexpected excluded STT values: {sorted(excluded_stt)}")
    labeled = review["annotation_status"].eq("labeled")
    if not review.loc[labeled, "news_type"].isin(VALID_LABELS).all():
        raise ValueError("Every labeled review row must contain one valid News Type")
    return review


def _build_resolution(review: pd.DataFrame) -> pd.DataFrame:
    labeled = review["annotation_status"].eq("labeled")
    resolution = pd.DataFrame({
        "review_batch": REVIEW_BATCH,
        "applied_date": REVIEW_APPLIED_DATE,
        "stt": review["STT"],
        "news_id": review["news_id"],
        "previous_news_type": review["previous_news_type"],
        "previous_annotation_status": "review_required",
        "final_news_type": review["news_type"].where(labeled, ""),
        "final_annotation_status": review["annotation_status"],
        "exclusion_reason": review["STT"].map(EXCLUSION_REASONS).fillna(""),
        "annotator_note": review["annotator_note"],
        "review_reason": review["review_reason"],
        "reviewed_content_sha256": review["content"].map(
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        ),
    }, columns=CORE_RESOLUTION_COLUMNS)
    return resolution


def _read_preserved_resolution(
    resolution_path: Path,
    expected_core: pd.DataFrame,
    expected_sha256: str,
) -> pd.DataFrame:
    """Validate and preserve the reviewed 16-column label/disposition artifact."""
    if not resolution_path.is_file():
        raise FileNotFoundError(f"Missing reviewed resolution: {resolution_path}")
    actual_hash = _sha256(resolution_path)
    if actual_hash != expected_sha256:
        raise ValueError(
            "Review38 resolution SHA-256 mismatch: "
            f"expected {expected_sha256}, received {actual_hash}"
        )
    resolution = pd.read_csv(
        resolution_path, dtype=str, keep_default_na=False,
    )
    if resolution.columns.tolist() != RESOLUTION_COLUMNS:
        raise ValueError(
            "Review38 resolution must contain the 12 label-decision columns followed "
            "by the four V2 disposition columns"
        )
    if len(resolution) != 38 or not resolution["news_id"].is_unique:
        raise ValueError("Review38 resolution must contain 38 unique article decisions")

    observed_core = resolution[CORE_RESOLUTION_COLUMNS].copy()
    rebuilt_core = expected_core[CORE_RESOLUTION_COLUMNS].copy()
    for column in CORE_RESOLUTION_COLUMNS:
        observed_core[column] = observed_core[column].map(_clean_text)
        rebuilt_core[column] = rebuilt_core[column].map(_clean_text)
    if not observed_core.equals(rebuilt_core):
        raise ValueError("Review38 resolution label decisions differ from the locked workbook")

    action_counts = resolution["v2_action"].value_counts().to_dict()
    if action_counts != EXPECTED_V2_ACTION_COUNTS:
        raise ValueError(f"Unexpected review38 V2 disposition counts: {action_counts}")

    for row in resolution.itertuples(index=False):
        matched_split = _clean_text(row.matched_split)
        matched_news_id = _clean_text(row.matched_news_id)
        reason = _clean_text(row.v2_reason)
        if row.v2_action == "TRAIN_V2":
            valid = not matched_split and not matched_news_id and reason == "NO_OVERLAP"
        elif row.v2_action == "HELDOUT_OVERLAP":
            valid = (
                matched_split in {"validation", "test"}
                and bool(matched_news_id)
                and reason == "SAME_EVENT_AS_HELDOUT"
            )
        elif row.v2_action == "DROP_DUPLICATE":
            valid = (
                matched_split in {"train", "review38"}
                and bool(matched_news_id)
                and reason == "NEAR_DUPLICATE"
            )
        elif row.v2_action == "EXCLUDED_SOURCE":
            valid = (
                not matched_split
                and not matched_news_id
                and reason == _clean_text(row.exclusion_reason)
                and row.final_annotation_status == "excluded"
            )
        else:
            valid = False
        if not valid:
            raise ValueError(f"Invalid V2 disposition at review38 STT {row.stt}")
    return resolution


def _validate_duplicate_dispositions(
    resolution: pd.DataFrame,
    clean: pd.DataFrame,
    duplicate_log: pd.DataFrame,
) -> None:
    """Require every dropped review alias to resolve to its kept clean article."""
    clean_ids = set(clean["news_id"])
    duplicate_rows = resolution.loc[resolution["v2_action"].eq("DROP_DUPLICATE")]
    for row in duplicate_rows.itertuples(index=False):
        if row.news_id in clean_ids:
            raise ValueError(f"Duplicate alias still exists in clean news: {row.news_id}")
        if row.matched_news_id not in clean_ids:
            raise ValueError(
                f"Approved duplicate representative is missing from clean news: {row.matched_news_id}"
            )
        matching_groups = []
        for group in duplicate_log.itertuples(index=False):
            merged_ids = set(str(group.merged_news_ids).split("|"))
            if row.news_id in merged_ids:
                matching_groups.append(group)
        if len(matching_groups) != 1:
            raise ValueError(f"Duplicate alias has invalid merge lineage: {row.news_id}")
        group = matching_groups[0]
        merged_ids = set(str(group.merged_news_ids).split("|"))
        if (
            group.kept_news_id != row.matched_news_id
            or getattr(group, "duplicate_reasons", "") != "approved_near_duplicate"
            or merged_ids != {row.news_id, row.matched_news_id}
            or str(group.rows_in_group) != "2"
            or str(group.rows_removed) != "1"
        ):
            raise ValueError(f"Duplicate alias lineage disagrees with review38: {row.news_id}")


def _apply_resolution(
    master: pd.DataFrame, review: pd.DataFrame, resolution: pd.DataFrame,
) -> pd.DataFrame:
    required = set(NEWS_COLUMNS + ANNOTATION_COLUMNS)
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(f"Annotation master is missing columns: {sorted(missing)}")
    if not master["news_id"].is_unique or master["news_id"].map(_clean_text).eq("").any():
        raise ValueError("Annotation master news_id values must be unique and nonempty")
    if not set(review["news_id"]).issubset(set(master["news_id"])):
        raise ValueError("A reviewed news_id does not exist in the annotation master")

    positions = {news_id: index for index, news_id in enumerate(master["news_id"])}
    updated = master.copy(deep=True)
    for review_row, final_row in zip(
        review.itertuples(index=False), resolution.itertuples(index=False), strict=True,
    ):
        index = positions[review_row.news_id]
        for column in ARTICLE_COLUMNS:
            if _clean_text(getattr(review_row, column)) != _clean_text(updated.at[index, column]):
                raise ValueError(f"Reviewed workbook and master differ in {column} at STT {review_row.STT}")

        current_type = _clean_text(updated.at[index, "news_type"])
        current_status = _clean_text(updated.at[index, "annotation_status"])
        current_note = _clean_text(updated.at[index, "annotator_note"])
        before_state = (
            _clean_text(final_row.previous_news_type),
            final_row.previous_annotation_status,
        )
        final_state = (final_row.final_news_type, final_row.final_annotation_status)
        if (current_type, current_status) == before_state:
            pass
        elif (current_type, current_status) == final_state and current_note == final_row.annotator_note:
            continue
        else:
            raise ValueError(
                f"Annotation master has a conflicting decision at STT {review_row.STT}: "
                f"{(current_type, current_status)!r}"
            )
        updated.at[index, "news_type"] = final_row.final_news_type
        updated.at[index, "annotation_status"] = final_row.final_annotation_status
        updated.at[index, "annotator_note"] = final_row.annotator_note
    return updated


def _baseline_model_rows(
    output_path: Path,
    metadata_path: Path,
    label2id: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not output_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Canonical baseline predictions and metadata must already exist")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    selected_run = metadata.get("selected_model_run", metadata.get("source_run"))
    if selected_run != BASELINE_RUN:
        raise ValueError(f"Canonical predictions are not bound to baseline run {BASELINE_RUN}")
    recorded_output_hash = metadata.get("output_csv_sha256")
    if recorded_output_hash != _sha256(output_path):
        raise ValueError("Canonical predictions do not match their current metadata hash")
    output = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    if output.columns.tolist() not in (OUTPUT_COLUMNS, FINAL_OUTPUT_COLUMNS):
        raise ValueError("Canonical prediction table has an unexpected schema")
    if metadata.get("output_columns") != output.columns.tolist():
        raise ValueError("Canonical prediction columns do not match their metadata")
    model_rows = output.loc[
        output["label_source"].eq("model"), PREDICTION_COLUMNS,
    ].copy()
    if len(model_rows) != EXPECTED_COUNTS["model"]:
        raise ValueError(f"Expected exactly 2,260 unchanged baseline model rows, got {len(model_rows)}")
    if not model_rows["news_id"].is_unique or model_rows["news_id"].eq("").any():
        raise ValueError("Baseline model rows must have unique, nonempty news_id values")
    if not model_rows["news_type"].isin(label2id).all():
        raise ValueError("Baseline model rows contain an unknown News Type")
    scores = pd.to_numeric(model_rows["prediction_score"], errors="raise")
    if scores.isna().any() or not scores.between(0, 1, inclusive="both").all():
        raise ValueError("Baseline prediction scores must be finite numbers between zero and one")
    return model_rows, metadata


def _project_paths(project_dir: Path) -> dict[str, Path]:
    review_dir = project_dir / "data/annotation/news_type/review38"
    return {
        "workbook": review_dir / "review38_source.xlsx",
        "resolution": review_dir / "review38_resolution.csv",
        "master": project_dir / "data/annotation/news_type/news_manually_labeled.csv",
        "clean": project_dir / "data/processed/news_clean.csv",
        "excluded": project_dir / "data/interim/news_excluded_records.csv",
        "duplicates": project_dir / "data/interim/duplicate_merge_log.csv",
        "mapping": project_dir / "models/news_type_phobert_cnn/label_mapping.json",
        "output": project_dir / "data/processed/news_type_predictions.csv",
        "metadata": project_dir / "data/processed/news_type_predictions_metadata.json",
    }


def finalize_news_type_review(
    project_dir: Path,
    *,
    expected_workbook_sha256: str = EXPECTED_WORKBOOK_SHA256,
    expected_resolution_sha256: str = EXPECTED_RESOLUTION_SHA256,
) -> dict[str, Any]:
    """Apply the human review and atomically publish one canonical output."""
    project_dir = Path(project_dir).resolve()
    paths = _project_paths(project_dir)
    review = _read_review(paths["workbook"], expected_workbook_sha256)
    expected_core_resolution = _build_resolution(review)
    resolution = _read_preserved_resolution(
        paths["resolution"], expected_core_resolution, expected_resolution_sha256,
    )

    master = pd.read_csv(paths["master"], dtype=str, keep_default_na=False)
    updated_master = _apply_resolution(master, review, resolution)
    clean = pd.read_csv(paths["clean"], dtype=str, keep_default_na=False)
    excluded_records = pd.read_csv(paths["excluded"], dtype=str, keep_default_na=False)
    duplicate_log = pd.read_csv(paths["duplicates"], dtype=str, keep_default_na=False)
    _validate_duplicate_dispositions(resolution, clean, duplicate_log)
    mapping = json.loads(paths["mapping"].read_text(encoding="utf-8-sig"))
    label2id = mapping["label2id"]
    if set(label2id) != VALID_LABELS:
        raise ValueError("The saved label mapping does not contain the expected seven News Types")

    baseline_rows, source_metadata = _baseline_model_rows(
        paths["output"], paths["metadata"], label2id,
    )
    plan, inference_inputs, _ = build_inference_plan(
        clean, updated_master, excluded_records, duplicate_log, label2id,
    )
    plan_counts = plan["label_source"].value_counts().to_dict()
    expected_plan_counts = {
        "manual": EXPECTED_COUNTS["manual"],
        "unlabeled": EXPECTED_COUNTS["model"],
        "excluded": EXPECTED_COUNTS["excluded"],
    }
    if plan_counts != expected_plan_counts:
        raise ValueError(f"Unexpected post-review inference partition: {plan_counts}")
    if set(inference_inputs["news_id"]) != set(baseline_rows["news_id"]):
        raise ValueError("Unchanged baseline model rows do not exactly cover current model targets")

    assignments = combine_news_type_predictions(plan, baseline_rows, label2id)
    final_counts = assignments["label_source"].value_counts().to_dict()
    if final_counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected final label-source counts: {final_counts}")
    if len(assignments) != len(clean) or assignments["news_id"].tolist() != clean["news_id"].tolist():
        raise ValueError("Canonical output must preserve every clean news_id exactly once and in order")

    # combine_news_type_predictions validates scores numerically. Restore the
    # original strings afterwards so accepted baseline row values never drift.
    source_by_id = baseline_rows.set_index("news_id", drop=False)
    model_mask = assignments["label_source"].eq("model")
    model_ids = assignments.loc[model_mask, "news_id"].tolist()
    assignments.loc[model_mask, "news_type"] = source_by_id.loc[model_ids, "news_type"].to_numpy()
    assignments.loc[model_mask, "prediction_score"] = source_by_id.loc[
        model_ids, "prediction_score"
    ].to_numpy()
    excluded_mask = assignments["label_source"].eq("excluded")
    if not assignments.loc[excluded_mask, ["news_type", "prediction_score"]].eq("").all().all():
        raise ValueError("Excluded rows must have blank News Type and prediction score")
    manual_mask = assignments["label_source"].eq("manual")
    if not assignments.loc[manual_mask, "prediction_score"].eq("").all():
        raise ValueError("Manual reference labels must not contain model scores")
    final_output = join_news_type_output(clean, assignments)
    if not final_output[NEWS_COLUMNS].equals(clean[NEWS_COLUMNS]):
        raise AssertionError("Publishing the joined output changed clean article fields")

    workbook_bytes = paths["workbook"].read_bytes()
    # Keep the reviewed 16-column artifact byte-identical.  It is an input to
    # the trained V2 provenance, not a generated side effect of finalization.
    resolution_bytes = paths["resolution"].read_bytes()
    master_bytes = _csv_bytes(updated_master)
    output_bytes = _csv_bytes(final_output)
    baseline_rows_bytes = _csv_bytes(baseline_rows.reset_index(drop=True))
    relative = lambda path: path.relative_to(project_dir).as_posix()
    input_hashes = {
        relative(paths["workbook"]): _sha256_bytes(workbook_bytes),
        relative(paths["resolution"]): _sha256_bytes(resolution_bytes),
        relative(paths["master"]): _sha256_bytes(master_bytes),
        relative(paths["clean"]): _sha256(paths["clean"]),
        relative(paths["excluded"]): _sha256(paths["excluded"]),
        relative(paths["duplicates"]): _sha256(paths["duplicates"]),
        relative(paths["mapping"]): _sha256(paths["mapping"]),
    }
    metadata = {
        "schema_version": "news_type_predictions_v3",
        "published_date": REVIEW_APPLIED_DATE,
        "source_run": BASELINE_RUN,
        "selected_model_unchanged": True,
        "checkpoint_epoch": source_metadata.get("checkpoint_epoch"),
        "checkpoint_sha256": source_metadata.get("checkpoint_sha256"),
        "test_manifest_sha256": source_metadata.get("test_manifest_sha256"),
        "label_mapping": mapping,
        "preprocessing": source_metadata.get("preprocessing", {}),
        "review_batch": {
            "name": REVIEW_BATCH,
            "source_workbook": relative(paths["workbook"]),
            "source_workbook_sha256": expected_workbook_sha256,
            "resolution": relative(paths["resolution"]),
            "resolution_sha256": _sha256_bytes(resolution_bytes),
            "reviewed_rows": 38,
            "manual_labels": 34,
            "canonical_review_articles": 32,
            "duplicate_aliases": 2,
            "excluded": 4,
        },
        "baseline_model_rows": {
            "count": EXPECTED_COUNTS["model"],
            "sha256": _sha256_bytes(baseline_rows_bytes),
            "preserved_without_reinference": True,
        },
        "input_file_sha256": input_hashes,
        "label_source_counts": EXPECTED_COUNTS,
        "output_rows": len(final_output),
        "output_columns": final_output.columns.tolist(),
        "output_csv_sha256": _sha256_bytes(output_bytes),
        "excluded_news_ids": final_output.loc[excluded_mask, "news_id"].tolist(),
        "score_description": (
            "Uncalibrated maximum Softmax score from the unchanged selected baseline; "
            "blank for manual and excluded rows."
        ),
    }
    metadata_bytes = _json_bytes(metadata)

    # All computation and validation finishes before the first write.
    _atomic_write_if_changed(paths["master"], master_bytes)
    _atomic_write_if_changed(paths["output"], output_bytes)
    _atomic_write_if_changed(paths["metadata"], metadata_bytes)
    return {
        "output": relative(paths["output"]),
        "metadata": relative(paths["metadata"]),
        "resolution": relative(paths["resolution"]),
        "counts": EXPECTED_COUNTS.copy(),
        "output_csv_sha256": metadata["output_csv_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    result = finalize_news_type_review(args.project_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
