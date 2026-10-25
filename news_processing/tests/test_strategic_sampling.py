"""Checks for annotation sampling that must not invent or erase human labels."""

import pandas as pd
import pytest

from src.annotation_utils import ANNOTATION_COLUMNS, NEWS_COLUMNS
from src.strategic_sampling import (
    allocate_source_quotas,
    build_strategic_annotation_sample,
)


TOPICS = [
    ("governance_capital", "Hội đồng quản trị bổ nhiệm tổng giám đốc, chia cổ tức"),
    ("financial_results", "Báo cáo tài chính: doanh thu và lợi nhuận quý tăng trưởng"),
    ("business_operations", "Doanh nghiệp ký hợp đồng và đầu tư mở rộng nhà máy"),
    ("legal_compliance", "Doanh nghiệp bị xử phạt thuế sau thanh tra vi phạm"),
    ("industry", "Triển vọng ngành thủy sản và giá nguyên liệu toàn ngành"),
    ("macro", "Lạm phát CPI, tỷ giá và chính sách tiền tệ của Ngân hàng Nhà nước"),
    ("stock_market", "Giá cổ phiếu tăng, thanh khoản và khuyến nghị định giá"),
    ("mention_only", "Danh sách doanh nghiệp tham gia chương trình cộng đồng"),
]


@pytest.fixture
def annotation_inputs():
    source_names = (
        ["CafeF"] * 42 + ["VietNamNet"] * 24 + ["BNews"] * 16
        + ["CafeBiz"] * 10 + ["NguoiQuanSat"] * 5 + ["NhaDauTu"] * 3
    )
    rows = []
    for index, source in enumerate(source_names):
        _, topic = TOPICS[index % len(TOPICS)]
        # Include topic overlaps and unusual lengths instead of identical filler.
        title = f"{topic} — bản tin {index}"
        if index % 11 == 0:
            title += "; cổ phiếu tăng sau công bố lợi nhuận và thanh tra thuế"
        content = f"{topic}. " * (1 if index % 17 == 0 else 6 + index % 19)
        if index % 23 == 0:
            content *= 20
        rows.append({
            "news_id": f"news-{index:03d}",
            "source": source,
            "url": f"https://example.com/news/{index}",
            "title": title,
            "summary": f"{topic}. Cập nhật thông tin doanh nghiệp.",
            "content": content,
            "published_date": f"{2021 + index % 6}-{index % 12 + 1:02d}-{index % 28 + 1:02d}",
            "tickers": ["MWG", "FPT", "MSN", "PNJ", "IMP", "MWG, FPT"][index % 6],
        })
    full_data = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    existing = full_data.iloc[:12].copy()
    existing["news_type"] = [TOPICS[index % 8][0] for index in range(12)]
    existing["annotation_status"] = "labeled"
    existing["annotator_note"] = [f"Ghi chú của người gán {index}" for index in range(12)]
    existing.loc[[2, 9], ["news_type", "annotation_status"]] = ["", "review_required"]
    return full_data, existing


def test_addition_preserves_annotations_and_original_news(annotation_inputs):
    full_data, existing = annotation_inputs
    original_data = full_data.copy(deep=True)
    original_annotations = existing.copy(deep=True)

    additional, combined, selection_log, _ = build_strategic_annotation_sample(
        full_data, existing, sample_size=40, random_seed=42,
    )

    assert len(additional) == 40
    assert len(combined) == 52
    assert list(additional.columns) == NEWS_COLUMNS + ANNOTATION_COLUMNS
    assert list(combined.columns) == NEWS_COLUMNS + ANNOTATION_COLUMNS
    assert additional["news_id"].is_unique
    assert combined["news_id"].is_unique
    assert set(additional["news_id"]).isdisjoint(existing["news_id"])
    assert additional[ANNOTATION_COLUMNS].fillna("").eq("").all().all()
    pd.testing.assert_frame_equal(combined.iloc[:12].reset_index(drop=True), existing)

    selected_ids = additional["news_id"].tolist()
    expected_news = full_data.set_index("news_id").loc[selected_ids].reset_index()
    pd.testing.assert_frame_equal(
        additional[NEWS_COLUMNS].reset_index(drop=True), expected_news[NEWS_COLUMNS],
    )
    assert set(selection_log["news_id"]) == set(selected_ids)
    pd.testing.assert_frame_equal(full_data, original_data)
    pd.testing.assert_frame_equal(existing, original_annotations)


def test_selection_is_stable_when_input_row_order_changes(annotation_inputs):
    full_data, existing = annotation_inputs
    first = build_strategic_annotation_sample(full_data, existing, sample_size=40)
    shuffled = build_strategic_annotation_sample(
        full_data.sample(frac=1, random_state=901), existing, sample_size=40,
    )

    for original_frame, repeated_frame in zip(first, shuffled):
        pd.testing.assert_frame_equal(original_frame, repeated_frame)


def test_source_quotas_are_exact_and_rare_hints_are_audit_only(annotation_inputs):
    full_data, existing = annotation_inputs
    existing = existing.copy()
    labeled_rows = existing["annotation_status"].eq("labeled")
    existing.loc[labeled_rows, "news_type"] = (
        ["financial_results"] * 4 + ["business_operations"] * 3
        + ["industry"] * 2 + ["macro"]
    )
    additional, _, selection_log, source_quotas = build_strategic_annotation_sample(
        full_data, existing, sample_size=40,
    )
    audit_columns = {
        "news_id", "selection_reason", "rare_label_score", "ambiguity_score",
        "anomaly_score", "priority_hints", "topic_cluster", "source", "tickers",
        "published_date",
    }
    assert audit_columns.issubset(selection_log.columns)
    assert selection_log["news_id"].is_unique
    assert selection_log["selection_reason"].fillna("").str.len().gt(0).all()
    assert selection_log["priority_hints"].fillna("").str.len().gt(0).any()
    scores = selection_log[["rare_label_score", "ambiguity_score", "anomaly_score"]]
    assert scores.notna().all().all()
    assert scores.ge(0).all().all()
    assert selection_log["topic_cluster"].notna().all()
    assert additional["news_type"].fillna("").eq("").all()

    quota_columns = {"source", "available_count", "target_count", "selected_count"}
    assert quota_columns.issubset(source_quotas.columns)
    quota_by_source = source_quotas.set_index("source")
    available = full_data.loc[~full_data["news_id"].isin(existing["news_id"])]
    counts = available["source"].value_counts().sort_index()
    assert set(quota_by_source.index) == set(counts.index)
    assert quota_by_source["available_count"].to_dict() == counts.to_dict()
    assert int(quota_by_source["target_count"].sum()) == 40
    assert quota_by_source["target_count"].ge(1).all()
    assert quota_by_source["target_count"].le(quota_by_source["available_count"]).all()
    assert quota_by_source["target_count"].eq(quota_by_source["selected_count"]).all()
    actual = additional["source"].value_counts().reindex(quota_by_source.index, fill_value=0)
    assert actual.to_dict() == quota_by_source["selected_count"].to_dict()


def test_source_quota_cap_relaxes_when_available_sources_cannot_fill_it():
    counts = pd.Series({"Dominant": 900, "Second": 100, "Third": 40, "Tiny": 10})
    quotas = allocate_source_quotas(counts, sample_size=100, max_share=0.2)

    assert quotas.index.tolist() == sorted(counts.index)
    assert pd.api.types.is_integer_dtype(quotas.dtype)
    assert int(quotas.sum()) == 100
    assert quotas.ge(1).all()
    assert quotas.le(counts.reindex(quotas.index)).all()
    assert quotas.max() > 20
    pd.testing.assert_series_equal(
        quotas, allocate_source_quotas(counts.iloc[::-1], sample_size=100, max_share=0.2),
    )
