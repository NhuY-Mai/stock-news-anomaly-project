import hashlib

import pandas as pd
import pytest

from src.annotation_utils import (
    ANNOTATION_COLUMNS,
    NEWS_COLUMNS,
    GOLD_PRICE_LABEL_RULE,
    COMPANY_AWARDS_LISTING_RULE,
    APPROVED_COMPANY_REVIEW_LABELS,
    REVIEWED_CONTENT_ARTICLE_ID,
    REVIEWED_CONTENT_SHA256,
    apply_approved_label_corrections,
    build_annotation_sample,
    extend_annotation_sample,
    normalize_news_type,
    refresh_annotation_content,
    save_annotation_csv,
    summarize_sample_coverage,
)


def make_news_data(row_count: int = 80) -> pd.DataFrame:
    sources = ["SourceA", "SourceB", "SourceC", "SourceD"]
    tickers = ["AAA", "BBB", "CCC", "AAA, BBB"]
    rows = []
    for index in range(row_count):
        year = 2021 + index % 4
        rows.append(
            {
                "news_id": f"news-{index:03d}",
                "source": sources[index % len(sources)],
                "url": f"https://example.com/{index}",
                "title": f"Title {index}",
                "summary": f"Summary {index}",
                "content": "content " * (30 + index),
                "published_date": f"{year}-01-{index % 28 + 1:02d}",
                "tickers": tickers[index % len(tickers)],
            }
        )
    return pd.DataFrame(rows)


def test_annotation_sample_is_deterministic_and_unique():
    data = make_news_data()

    first_sample = build_annotation_sample(data, sample_size=40, random_seed=42)
    second_sample = build_annotation_sample(data, sample_size=40, random_seed=42)

    assert first_sample["news_id"].tolist() == second_sample["news_id"].tolist()
    assert len(first_sample) == 40
    assert first_sample["news_id"].is_unique
    assert set(first_sample["sampling_partition"]) == {
        "representative",
        "coverage_boost",
    }


def test_annotation_sample_preserves_metadata_coverage():
    data = make_news_data()
    sample = build_annotation_sample(data, sample_size=40, random_seed=42)
    report = summarize_sample_coverage(data, sample)

    assert sample["source"].nunique() == data["source"].nunique()
    assert sample["sampling_year"].nunique() == 4
    assert set(sample["content_length_group"]) == {
        "short",
        "medium",
        "long",
        "very_long",
    }
    assert report["sample_count"].gt(0).all()


def test_extension_keeps_existing_labels_and_excludes_previous_articles():
    data = make_news_data()
    existing = data.iloc[:10].copy()
    existing["news_type"] = "financial_results"
    existing["annotation_status"] = "labeled"
    existing["annotator_note"] = "Keep my annotation"
    existing.loc[0, ["news_type", "annotation_status"]] = ["", "review_required"]

    new, combined, lineage, _ = extend_annotation_sample(
        data, existing, target_size=40, priority_quotas={},
    )
    pd.testing.assert_frame_equal(combined.iloc[:10], existing, check_dtype=False)
    assert len(new) == 30
    assert combined["news_id"].is_unique
    assert set(new["news_id"]).isdisjoint(existing["news_id"])
    assert new[ANNOTATION_COLUMNS].eq("").all().all()
    assert list(combined.columns) == NEWS_COLUMNS + ANNOTATION_COLUMNS
    assert len(lineage) == len(new)

    _, repeated, _, _ = extend_annotation_sample(
        data.sample(frac=1, random_state=99), existing,
        target_size=40, priority_quotas={},
    )
    assert repeated["news_id"].tolist() == combined["news_id"].tolist()


def test_candidate_shortage_is_filled_without_inventing_labels():
    data = make_news_data()
    data.loc[20:21, "title"] = "Chính sách tiền tệ và lạm phát"
    existing = data.iloc[:10].copy()
    for column in ANNOTATION_COLUMNS:
        existing[column] = ""

    new, combined, lineage, availability = extend_annotation_sample(
        data, existing, target_size=40, priority_quotas={"macro": 10},
    )
    assert len(combined) == 40
    assert availability.iloc[0]["available_at_selection"] == 2
    assert availability.iloc[0]["selected_count"] == 2
    assert lineage["selection_method"].value_counts().to_dict() == {
        "year_stratified_random": 28, "keyword_candidates": 2,
    }
    assert new["news_type"].eq("").all()


def test_export_refuses_to_overwrite_new_human_annotations(tmp_path):
    data = make_news_data(8)
    for column in ANNOTATION_COLUMNS:
        data[column] = ""
    output_path = tmp_path / "annotations.csv"
    save_annotation_csv(data, output_path)
    labeled = data.copy()
    labeled.loc[0, ["news_type", "annotation_status"]] = ["macro", "labeled"]
    save_annotation_csv(labeled, output_path)

    with pytest.raises(FileExistsError, match="overwrite annotations"):
        save_annotation_csv(data, output_path)
    saved = pd.read_csv(output_path, keep_default_na=False)
    assert saved.loc[0, "news_type"] == "macro"
    assert saved.loc[0, "annotation_status"] == "labeled"


def test_content_refresh_preserves_human_fields_and_quarantined_articles():
    annotations = make_news_data(3)
    annotations["news_type"] = "financial_results"
    annotations["annotation_status"] = ["labeled", "review_required", "labeled"]
    annotations["annotator_note"] = ["Ghi chú giữ nguyên", "Cần lấy lại bài", "Nhãn cũ"]
    annotations.loc[0, "tickers"] = "MCH"
    clean = annotations.iloc[[0, 2]][NEWS_COLUMNS].copy()
    clean.loc[0, "content"] = "Thân bài đã bỏ footer."
    excluded = pd.DataFrame([{
        "news_id": "news-001", "content": "Danh sách tin khác, không phải thân bài.",
        "exclusion_reason": "CONTENT_QUALITY:MISSING_ARTICLE_BODY",
    }])
    merges = pd.DataFrame(columns=["kept_news_id", "merged_news_ids"])

    refreshed, audit = refresh_annotation_content(annotations, clean, excluded, merges)

    protected = annotations.columns.difference(["content"])
    pd.testing.assert_frame_equal(refreshed[protected], annotations[protected])
    assert refreshed.loc[0, "content"] == "Thân bài đã bỏ footer."
    assert refreshed.loc[1, "content"] == ""
    assert len(refreshed) == 3
    assert audit["training_eligible"].tolist() == [True, False, True]


def test_content_refresh_maps_duplicate_ids_and_blocks_conflicting_labels():
    annotations = make_news_data(2)
    annotations["news_type"] = ["financial_results", "stock_market"]
    annotations["annotation_status"] = "labeled"
    annotations["annotator_note"] = "Không thay đổi"
    clean = annotations.iloc[[0]][NEWS_COLUMNS]
    excluded = pd.DataFrame(columns=["news_id", "content", "exclusion_reason"])
    merges = pd.DataFrame([{
        "kept_news_id": "news-000", "merged_news_ids": "news-000|news-001",
    }])

    refreshed, audit = refresh_annotation_content(annotations, clean, excluded, merges)

    assert refreshed["news_id"].tolist() == annotations["news_id"].tolist()
    assert refreshed[ANNOTATION_COLUMNS].equals(annotations[ANNOTATION_COLUMNS])
    assert refreshed["content"].nunique() == 1
    assert not audit["training_eligible"].any()
    assert audit["content_issue"].eq("DUPLICATE_LABEL_CONFLICT").all()


def test_content_refresh_does_not_admit_duplicate_training_articles():
    annotations = make_news_data(2)
    annotations["news_type"] = "financial_results"
    annotations["annotation_status"] = "labeled"
    annotations["annotator_note"] = ""
    excluded = pd.DataFrame(columns=["news_id", "content", "exclusion_reason"])
    merges = pd.DataFrame([{
        "kept_news_id": "news-000", "merged_news_ids": "news-000|news-001",
    }])
    _, audit = refresh_annotation_content(
        annotations, annotations.iloc[[0]][NEWS_COLUMNS], excluded, merges,
    )
    assert audit["training_eligible"].tolist() == [True, False]
    assert audit.loc[1, "content_issue"] == "DUPLICATE_ARTICLE"


def test_content_refresh_rejects_ids_without_lineage():
    annotations = make_news_data(2)
    for column in ANNOTATION_COLUMNS:
        annotations[column] = ""
    with pytest.raises(ValueError, match="no cleaning lineage"):
        refresh_annotation_content(
            annotations, annotations.iloc[[0]][NEWS_COLUMNS],
            pd.DataFrame(columns=["news_id", "content", "exclusion_reason"]),
            pd.DataFrame(columns=["kept_news_id", "merged_news_ids"]),
        )


def test_content_refresh_recovers_body_from_valid_crawl_of_same_url():
    annotations = make_news_data(1)
    annotations["news_type"] = "financial_results"
    annotations["annotation_status"] = "labeled"
    annotations["annotator_note"] = "Giữ nhãn đã nhập"
    valid_crawl = annotations[NEWS_COLUMNS].copy()
    valid_crawl.loc[0, "news_id"] = "alternate-crawl"
    valid_crawl.loc[0, "content"] = "Nội dung đầy đủ từ bản crawl hợp lệ cùng URL."
    refreshed, audit = refresh_annotation_content(
        annotations, valid_crawl,
        pd.DataFrame([{
            "news_id": "news-000", "content": "Tin phụ",
            "exclusion_reason": "CONTENT_QUALITY:MISSING_ARTICLE_BODY",
        }]),
        pd.DataFrame(columns=["kept_news_id", "merged_news_ids"]),
    )
    assert refreshed.loc[0, "news_id"] == "news-000"
    assert refreshed.loc[0, "content"] == valid_crawl.loc[0, "content"]
    assert audit.loc[0, "canonical_news_id"] == "alternate-crawl"
    assert bool(audit.loc[0, "training_eligible"])


def test_normalize_news_type_merges_macro_into_industry():
    annotations = make_news_data(2)
    annotations["news_type"] = ["macro", "industry"]
    annotations["annotation_status"] = "labeled"
    annotations["annotator_note"] = ""
    clean = annotations[NEWS_COLUMNS].copy()
    excluded = pd.DataFrame(columns=["news_id", "content", "exclusion_reason"])
    merges = pd.DataFrame(columns=["kept_news_id", "merged_news_ids"])

    refreshed, audit = refresh_annotation_content(annotations, clean, excluded, merges)

    assert normalize_news_type("macro") == "industry"
    assert refreshed["news_type"].tolist() == ["macro", "industry"]
    assert audit["training_news_type"].tolist() == ["industry", "industry"]
    assert audit["training_eligible"].all()


def _gold_correction_fixture():
    annotations = make_news_data(3)
    annotations["news_type"] = ["mention_only", "financial_results", "mention_only"]
    annotations["annotation_status"] = "labeled"
    annotations["annotator_note"] = ["Keep note", "Keep earnings", "Unreviewed gold title"]
    annotations.loc[2, "title"] = "Gold price bulletin, not in the approved list"
    corrections = pd.DataFrame([{
        "news_id": "news-000", "old_news_type": "mention_only", "new_news_type": "industry",
        "rule_id": GOLD_PRICE_LABEL_RULE,
        "content_sha256": hashlib.sha256(annotations.loc[0, "content"].encode("utf-8")).hexdigest(),
    }])
    return annotations, corrections


def test_approved_label_correction_preserves_unrelated_rows_fields_and_order():
    annotations, corrections = _gold_correction_fixture()
    original = annotations.copy(deep=True)
    corrected = apply_approved_label_corrections(annotations, corrections)
    expected = annotations.copy(deep=True)
    expected.loc[0, "news_type"] = "industry"
    pd.testing.assert_frame_equal(corrected, expected)
    pd.testing.assert_frame_equal(annotations, original)
    pd.testing.assert_frame_equal(
        apply_approved_label_corrections(corrected, corrections), corrected,
    )


@pytest.mark.parametrize("case", [
    "unknown_id", "duplicate_correction", "duplicate_annotation", "empty_id",
    "wrong_rule", "wrong_old_label", "wrong_new_label", "stale_label",
    "stale_content", "unreviewed_status", "empty_content", "missing_audit_column",
])
def test_approved_label_correction_rejects_unapproved_or_stale_changes(case):
    annotations, corrections = _gold_correction_fixture()
    if case == "unknown_id":
        corrections.loc[0, "news_id"] = "unknown"
    elif case == "duplicate_correction":
        corrections = pd.concat([corrections, corrections], ignore_index=True)
    elif case == "duplicate_annotation":
        annotations.loc[1, "news_id"] = annotations.loc[0, "news_id"]
    elif case == "empty_id":
        corrections.loc[0, "news_id"] = " "
    elif case == "wrong_rule":
        corrections.loc[0, "rule_id"] = "other_rule"
    elif case == "wrong_old_label":
        corrections.loc[0, "old_news_type"] = "financial_results"
    elif case == "wrong_new_label":
        corrections.loc[0, "new_news_type"] = "stock_market"
    elif case == "stale_label":
        annotations.loc[0, "news_type"] = "business_operations"
    elif case == "stale_content":
        annotations.loc[0, "content"] += " New body"
    elif case == "unreviewed_status":
        annotations.loc[0, "annotation_status"] = "review_required"
    elif case == "empty_content":
        annotations.loc[0, "content"] = ""
    elif case == "missing_audit_column":
        corrections = corrections.drop(columns="content_sha256")
    original = annotations.copy(deep=True)
    with pytest.raises(ValueError):
        apply_approved_label_corrections(annotations, corrections)
    pd.testing.assert_frame_equal(annotations, original)


def test_empty_approved_correction_table_is_a_noop():
    annotations, corrections = _gold_correction_fixture()
    corrected = apply_approved_label_corrections(annotations, corrections.iloc[:0])
    pd.testing.assert_frame_equal(corrected, annotations)


def _company_review_correction_fixture(news_id=None):
    news_id = news_id or next(iter(APPROVED_COMPANY_REVIEW_LABELS))
    old_label, new_label = APPROVED_COMPANY_REVIEW_LABELS[news_id]
    annotations = make_news_data(3)
    annotations.loc[0, "news_id"] = news_id
    annotations["news_type"] = [old_label, "financial_results", "industry"]
    annotations["annotation_status"] = "labeled"
    annotations["annotator_note"] = ["Keep reviewed note", "Keep financial note", "Keep industry note"]
    corrections = pd.DataFrame([{
        "news_id": news_id, "old_news_type": old_label, "new_news_type": new_label,
        "rule_id": COMPANY_AWARDS_LISTING_RULE,
        "content_sha256": hashlib.sha256(annotations.loc[0, "content"].encode("utf-8")).hexdigest(),
        "reviewed_content_sha256": "",
    }])
    return annotations, corrections


@pytest.mark.parametrize("news_id", list(APPROVED_COMPANY_REVIEW_LABELS))
def test_company_review_changes_only_each_approved_article_label(news_id):
    annotations, corrections = _company_review_correction_fixture(news_id)
    original = annotations.copy(deep=True)
    corrected = apply_approved_label_corrections(annotations, corrections)
    expected = original.copy(deep=True)
    expected.loc[0, "news_type"] = APPROVED_COMPANY_REVIEW_LABELS[news_id][1]
    pd.testing.assert_frame_equal(corrected, expected)
    pd.testing.assert_frame_equal(annotations, original)
    pd.testing.assert_frame_equal(apply_approved_label_corrections(corrected, corrections), corrected)


@pytest.mark.parametrize("case", ["unapproved_id", "wrong_old_label", "wrong_new_label", "wrong_rule", "stale_label", "stale_content"])
def test_company_review_rejects_unapproved_or_stale_revisions(case):
    annotations, corrections = _company_review_correction_fixture()
    if case == "unapproved_id":
        annotations.loc[0, "news_id"] = corrections.loc[0, "news_id"] = "not-an-approved-article"
    elif case == "wrong_old_label":
        corrections.loc[0, "old_news_type"] = "industry"
    elif case == "wrong_new_label":
        corrections.loc[0, "new_news_type"] = "financial_results"
    elif case == "wrong_rule":
        corrections.loc[0, "rule_id"] = GOLD_PRICE_LABEL_RULE
    elif case == "stale_label":
        annotations.loc[0, "news_type"] = "legal_compliance"
    else:
        annotations.loc[0, "content"] += " Newly edited body"
    original = annotations.copy(deep=True)
    with pytest.raises(ValueError):
        apply_approved_label_corrections(annotations, corrections)
    pd.testing.assert_frame_equal(annotations, original)


def test_reviewed_listing_fingerprint_allows_only_the_approved_clean_body(monkeypatch):
    annotations, corrections = _company_review_correction_fixture(REVIEWED_CONTENT_ARTICLE_ID)
    original_fingerprint = corrections.loc[0, "content_sha256"]
    annotations.loc[0, "content"] = "Approved cleaned listing article."
    corrections.loc[0, "reviewed_content_sha256"] = hashlib.sha256(
        annotations.loc[0, "content"].encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(
        "src.annotation_utils.REVIEWED_CONTENT_SHA256", corrections.loc[0, "reviewed_content_sha256"],
    )
    corrected = apply_approved_label_corrections(annotations, corrections)
    assert corrected.loc[0, "news_type"] == "governance_capital"
    assert corrections.loc[0, "content_sha256"] == original_fingerprint
    pd.testing.assert_frame_equal(apply_approved_label_corrections(corrected, corrections), corrected)
    corrected.loc[0, "content"] += " Unapproved extra text"
    with pytest.raises(ValueError, match="content changed"):
        apply_approved_label_corrections(corrected, corrections)


@pytest.mark.parametrize("fingerprint", ["not-a-hash", "f" * 63, "F" * 64, " " + "f" * 64])
def test_reviewed_listing_rejects_malformed_post_clean_fingerprints(fingerprint):
    annotations, corrections = _company_review_correction_fixture(REVIEWED_CONTENT_ARTICLE_ID)
    corrections.loc[0, "reviewed_content_sha256"] = fingerprint
    with pytest.raises(ValueError, match="post-clean"):
        apply_approved_label_corrections(annotations, corrections)


def test_reviewed_listing_fingerprint_is_pinned_to_the_actual_approved_body():
    assert REVIEWED_CONTENT_SHA256 == "eb2003c9d05d18bae8dbe7a83c0447fbbe7e3732d954e3403f31b449d278e0d6"
    annotations, corrections = _company_review_correction_fixture(REVIEWED_CONTENT_ARTICLE_ID)
    corrections.loc[0, "reviewed_content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="post-clean"):
        apply_approved_label_corrections(annotations, corrections)


def test_post_clean_fingerprint_is_not_allowed_for_another_company_article():
    annotations, corrections = _company_review_correction_fixture()
    corrections.loc[0, "reviewed_content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="post-clean"):
        apply_approved_label_corrections(annotations, corrections)


def test_post_clean_fingerprint_does_not_relax_legacy_gold_checks():
    annotations, corrections = _gold_correction_fixture()
    corrections["reviewed_content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="post-clean"):
        apply_approved_label_corrections(annotations, corrections)


def test_empty_post_clean_column_preserves_existing_gold_revision_behavior():
    annotations, corrections = _gold_correction_fixture()
    corrections["reviewed_content_sha256"] = ""
    corrected = apply_approved_label_corrections(annotations, corrections)
    assert corrected.loc[0, "news_type"] == "industry"

