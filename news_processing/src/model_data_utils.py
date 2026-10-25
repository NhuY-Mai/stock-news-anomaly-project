"""Utilities for preparing PhoBERT-CNN model inputs and stratified data splits."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import pandas as pd

from .annotation_utils import (
    _proportional_group_sample,
    refresh_annotation_content,
)
from .validation_utils import normalize_title


MODEL_SPLIT_COLUMNS = [
    "news_id",
    "canonical_news_id",
    "source",
    "url",
    "published_date",
    "tickers",
    "title",
    "summary",
    "content",
    "input_text",
    "news_type",
    "label_id",
]

WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def _clean_text_segment(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return WHITESPACE_RE.sub(" ", text).strip()


def _ensure_terminal_punctuation(text: str) -> str:
    if not text:
        return ""
    if text[-1] in ".!?…":
        return text
    return f"{text}."


def is_redundant_summary(title: object, summary: object, content: object) -> bool:
    """Skip empty summaries, title duplicates, or a heuristic match to the opening."""
    summary_text = _clean_text_segment(summary)
    if not summary_text:
        return True

    norm_summary = NON_WORD_RE.sub("", summary_text.casefold())
    if not norm_summary:
        return True

    title_text = _clean_text_segment(title)
    if title_text and norm_summary == NON_WORD_RE.sub("", title_text.casefold()):
        return True

    content_text = _clean_text_segment(content)
    if not content_text:
        return False

    # Compare the opening after removing punctuation and case differences.
    content_prefix = content_text[: max(len(summary_text) + 40, 250)]
    norm_content_prefix = NON_WORD_RE.sub("", content_prefix.casefold())
    # Allow a shortened prefix; this is not a test for verbatim duplication.
    comparable_summary = norm_summary[: max(20, len(norm_summary) - 6)]
    return norm_content_prefix.startswith(comparable_summary)


def build_phobert_input_text(title: object, summary: object, content: object) -> str:
    """Join title, optional summary and body without tokenizing or truncating."""
    title_text = _clean_text_segment(title)
    summary_text = _clean_text_segment(summary)
    content_text = _clean_text_segment(content)

    segments: list[str] = []
    if title_text:
        segments.append(_ensure_terminal_punctuation(title_text))
    if summary_text and not is_redundant_summary(title_text, summary_text, content_text):
        segments.append(_ensure_terminal_punctuation(summary_text))
    if content_text:
        segments.append(content_text)

    return " ".join(segments).strip()


def prepare_eligible_model_dataset(
    annotations: pd.DataFrame,
    clean_news: pd.DataFrame,
    excluded_records: pd.DataFrame,
    duplicate_merge_log: pd.DataFrame,
    label2id: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter training-eligible rows, map to 7 main classes, and build PhoBERT input_text."""
    refreshed, audit = refresh_annotation_content(
        annotations, clean_news, excluded_records, duplicate_merge_log
    )

    eligible_mask = audit["training_eligible"].astype(bool)
    eligible = refreshed.loc[eligible_mask].copy()
    eligible["canonical_news_id"] = audit.loc[eligible_mask, "canonical_news_id"].to_numpy()
    eligible["news_type"] = audit.loc[eligible_mask, "training_news_type"].to_numpy()

    unknown_labels = sorted(set(eligible["news_type"]).difference(label2id))
    if unknown_labels:
        raise ValueError(f"Unmapped training labels found: {unknown_labels}")

    eligible["label_id"] = eligible["news_type"].map(label2id).astype(int)
    eligible["summary_redundant"] = [
        is_redundant_summary(title, summary, content)
        for title, summary, content in zip(
            eligible["title"], eligible["summary"], eligible["content"]
        )
    ]
    eligible["input_text"] = [
        build_phobert_input_text(title, summary, content)
        for title, summary, content in zip(
            eligible["title"], eligible["summary"], eligible["content"]
        )
    ]

    eligible = eligible.sort_values("news_id", kind="stable").reset_index(drop=True)
    return eligible[MODEL_SPLIT_COLUMNS + ["summary_redundant"]], audit


def stratified_train_val_test_split(
    data: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_column: str = "label_id",
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into Stratified Train / Validation / Test sets deterministically."""
    if any(r <= 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("All split ratios must be positive")
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if group_column not in data.columns:
        raise ValueError(f"Missing stratification column: {group_column}")

    ordered = data.sort_values("news_id", kind="stable").set_index("news_id", drop=False)
    if not ordered.index.is_unique:
        raise ValueError("news_id must be unique before splitting")

    total_rows = len(ordered)
    test_size = round(total_rows * test_ratio)
    val_size = round(total_rows * val_ratio)
    train_size = total_rows - test_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Dataset is too small for the requested split ratios")

    test_split = _proportional_group_sample(
        ordered,
        group_column=group_column,
        sample_size=test_size,
        random_seed=random_seed + 10,
    ).copy()
    remaining_pool = ordered.drop(index=test_split.index).copy()

    val_split = _proportional_group_sample(
        remaining_pool,
        group_column=group_column,
        sample_size=val_size,
        random_seed=random_seed + 20,
    ).copy()
    train_split = remaining_pool.drop(index=val_split.index).copy()

    train_split = train_split.sample(frac=1, random_state=random_seed + 1).reset_index(drop=True)
    val_split = val_split.sample(frac=1, random_state=random_seed + 2).reset_index(drop=True)
    test_split = test_split.sample(frac=1, random_state=random_seed + 3).reset_index(drop=True)

    train_split["split"] = "train"
    val_split["split"] = "validation"
    test_split["split"] = "test"
    return train_split, val_split, test_split


def reuse_split_membership(
    data: pd.DataFrame,
    existing_splits: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refresh rows while retaining the exact saved split membership and order.

    Eligibility changes require an explicit decision; they never trigger a new
    random split here. Saved labels and text are not reused.
    """
    split_names = ("train", "validation", "test")
    if set(existing_splits) != set(split_names):
        raise ValueError("Existing splits must contain exactly train, validation, and test")
    missing_columns = set(MODEL_SPLIT_COLUMNS).difference(data.columns)
    if missing_columns:
        raise ValueError(f"Current data is missing model columns: {sorted(missing_columns)}")

    def check_ids(frame: pd.DataFrame, name: str) -> None:
        if "news_id" not in frame.columns:
            raise ValueError(f"Missing news_id in {name}")
        empty_ids = frame["news_id"].fillna("").astype(str).str.strip().eq("")
        if frame.empty or empty_ids.any() or not frame["news_id"].is_unique:
            raise ValueError(f"news_id must be nonempty and unique in {name}")

    check_ids(data, "current data")
    saved_ids = []
    for split_name in split_names:
        saved = existing_splits[split_name]
        check_ids(saved, f"saved {split_name}")
        if "split" not in saved.columns or not saved["split"].eq(split_name).all():
            raise ValueError(f"Invalid split tags in saved {split_name}")
        saved_ids.extend(saved["news_id"].tolist())

    if len(set(saved_ids)) != len(saved_ids):
        raise ValueError("Saved split memberships overlap")
    if set(saved_ids) != set(data["news_id"]):
        raise ValueError("Saved split IDs do not exactly cover current eligible data")

    current_by_id = data[MODEL_SPLIT_COLUMNS].set_index("news_id", drop=False)
    rebuilt_splits = []
    for split_name in split_names:
        ordered_ids = existing_splits[split_name]["news_id"].tolist()
        refreshed = current_by_id.loc[ordered_ids].copy().reset_index(drop=True)
        refreshed["split"] = split_name
        rebuilt_splits.append(refreshed)
    return tuple(rebuilt_splits)


def build_split_distribution_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label2id: dict[str, int],
    display_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize per-class sample counts and shares across train, validation, and test."""
    rows: list[dict[str, Any]] = []
    ordered_labels = [label for label, _ in sorted(label2id.items(), key=lambda item: item[1])]

    train_counts = train_df["news_type"].value_counts()
    val_counts = val_df["news_type"].value_counts()
    test_counts = test_df["news_type"].value_counts()

    for label in ordered_labels:
        train_count = int(train_counts.get(label, 0))
        validation_count = int(val_counts.get(label, 0))
        test_count = int(test_counts.get(label, 0))
        total_count = train_count + validation_count + test_count
        rows.append({
            "label_id": int(label2id[label]),
            "news_type": label,
            "display_name": (display_names or {}).get(label, label),
            "train_count": train_count,
            "val_count": validation_count,
            "test_count": test_count,
            "total_count": total_count,
            "train_pct": round(train_count / len(train_df) * 100, 2) if len(train_df) else 0.0,
            "val_pct": round(validation_count / len(val_df) * 100, 2) if len(val_df) else 0.0,
            "test_pct": round(test_count / len(test_df) * 100, 2) if len(test_df) else 0.0,
        })
    return pd.DataFrame(rows)


def compute_balanced_class_weights(
    train_df: pd.DataFrame,
    label2id: dict[str, int],
) -> pd.DataFrame:
    """Compute inverse-frequency balanced class weights strictly from the train split."""
    ordered_labels = [label for label, _ in sorted(label2id.items(), key=lambda item: item[1])]
    num_classes = len(ordered_labels)
    total_train = len(train_df)
    train_counts = train_df["news_type"].value_counts()

    rows: list[dict[str, Any]] = []
    for label in ordered_labels:
        count = int(train_counts.get(label, 0))
        if count <= 0:
            raise ValueError(f"Class '{label}' has 0 samples in train_df")
        weight = total_train / (num_classes * count)
        rows.append({
            "label_id": int(label2id[label]),
            "news_type": label,
            "train_count": count,
            "class_weight": round(float(weight), 4),
        })
    return pd.DataFrame(rows)


def verify_model_splits(
    full_eligible: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label2id: dict[str, int],
) -> pd.DataFrame:
    """Check integrity and exact-key overlap, not semantic or event independence."""
    splits = [("train", train_df), ("validation", val_df), ("test", test_df)]
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    def _pairwise_overlap_count(column: str, transform=None) -> int:
        sets = []
        for _, frame in splits:
            series = frame[column].fillna("").astype(str)
            if transform is not None:
                series = series.map(transform)
            sets.append({val for val in series if val})
        overlaps = (
            (sets[0] & sets[1])
            | (sets[0] & sets[2])
            | (sets[1] & sets[2])
        )
        return len(overlaps)

    required_cols = ["news_id", "input_text", "news_type", "label_id"]
    empty_required = int(
        combined[required_cols]
        .apply(lambda col: col.fillna("").astype(str).str.strip().eq(""))
        .any(axis=1)
        .sum()
    )

    expected_label_ids = set(label2id.values())
    missing_classes_count = sum(
        len(expected_label_ids.difference(set(pd.to_numeric(frame["label_id"], errors="coerce"))))
        for _, frame in splits
    )
    numeric_label_ids = pd.to_numeric(combined["label_id"], errors="coerce")
    invalid_label_id_count = int((~numeric_label_ids.isin(expected_label_ids)).sum())
    expected_row_labels = combined["news_type"].map(label2id)
    label_mapping_issue_count = int(expected_row_labels.ne(numeric_label_ids).sum())
    eligible_ids = set(full_eligible["news_id"].dropna())
    combined_ids = set(combined["news_id"].dropna())
    duplicate_ids_within_splits = sum(
        int(frame["news_id"].duplicated().sum()) for _, frame in splits
    )
    invalid_eligible_ids = int(
        full_eligible["news_id"].fillna("").astype(str).str.strip().eq("").sum()
        + full_eligible["news_id"].duplicated().sum()
    )

    checks = [
        {
            "check": "Đối soát tổng số mẫu (Train + Val + Test == Eligible)",
            "issue_count": abs(len(full_eligible) - len(combined)),
        },
        {
            "check": "Rò rỉ news_id giữa các tập",
            "issue_count": _pairwise_overlap_count("news_id"),
        },
        {
            "check": "Rò rỉ canonical_news_id giữa các tập",
            "issue_count": _pairwise_overlap_count("canonical_news_id"),
        },
        {
            "check": "Rò rỉ URL giữa các tập",
            "issue_count": _pairwise_overlap_count("url"),
        },
        {
            "check": "Trùng tiêu đề chuẩn hóa giữa các tập",
            "issue_count": _pairwise_overlap_count("title", transform=normalize_title),
        },
        {
            "check": "Trùng nội dung content giữa các tập",
            "issue_count": _pairwise_overlap_count("content"),
        },
        {
            "check": "Trường bắt buộc bị rỗng (input_text, news_type, label_id)",
            "issue_count": empty_required,
        },
        {
            "check": "Thiếu lớp nhãn ở bất kỳ tập nào (yêu cầu đủ 7/7 nhãn)",
            "issue_count": int(missing_classes_count),
        },
        {
            "check": "Mã số label_id ngoài phạm vi 0..6",
            "issue_count": invalid_label_id_count,
        },
        {
            "check": "news_id rỗng hoặc trùng trong dữ liệu đủ điều kiện",
            "issue_count": invalid_eligible_ids,
        },
        {
            "check": "news_id trùng trong cùng một tập",
            "issue_count": duplicate_ids_within_splits,
        },
        {
            "check": "news_id thiếu hoặc ngoài dữ liệu đủ điều kiện",
            "issue_count": len(eligible_ids.symmetric_difference(combined_ids)),
        },
        {
            "check": "news_type không khớp label_id",
            "issue_count": label_mapping_issue_count,
        },
    ]
    return pd.DataFrame(checks)

