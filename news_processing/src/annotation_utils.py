"""Utilities for building a reproducible news annotation sample."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd


CONTENT_LENGTH_LABELS = ["short", "medium", "long", "very_long"]

NEWS_COLUMNS = [
    "news_id", "source", "url", "title", "summary", "content",
    "published_date", "tickers",
]
ANNOTATION_COLUMNS = ["news_type", "annotation_status", "annotator_note"]
MERGED_NEWS_TYPE_LABELS = {"macro": "industry"}
GOLD_PRICE_LABEL_RULE = "gold_price_industry_v1"
COMPANY_AWARDS_LISTING_RULE = "company_awards_listing_v2"
APPROVED_COMPANY_REVIEW_LABELS = {
    "2243033b0d5b0474496fb850c24265ca1f2786be70d1f83f140db07746b4b16f": (
        "stock_market", "business_operations",
    ),
    "1d85806da71bbed30f87fb08768b156742ad9bb28ec7debb0b5f343c2eea0098": (
        "stock_market", "business_operations",
    ),
    "bcd95f1e82aea14779999077cfa9d1b50d67a61339ad9de7cab87876d69c5a97": (
        "financial_results", "business_operations",
    ),
    "24c79e1c2ffee2f35c6e081a2e24c9c9191cf492bbd051c8364f58ff364a9464": (
        "financial_results", "business_operations",
    ),
    "0924ce353ae49d611b69c2d14935624b6023ff60543e4208a02bdb20d6d65d62": (
        "stock_market", "governance_capital",
    ),
    "43b8ec1ee1eb43780236646e6a91148b484882fc0a51493f3892249d75c2f74d": (
        "stock_market", "governance_capital",
    ),
}
REVIEWED_CONTENT_ARTICLE_ID = "0924ce353ae49d611b69c2d14935624b6023ff60543e4208a02bdb20d6d65d62"
REVIEWED_CONTENT_SHA256 = "eb2003c9d05d18bae8dbe7a83c0447fbbe7e3732d954e3403f31b449d278e0d6"


def apply_approved_label_corrections(
    annotations: pd.DataFrame, corrections: pd.DataFrame,
) -> pd.DataFrame:
    """Apply reviewed article IDs only; never infer labels from title keywords.

    The audit retains each original label and body fingerprint. Reapplying an
    already completed correction is a no-op; unrelated human work stays intact.
    """
    annotation_columns = {"news_id", "news_type", "annotation_status", "content"}
    correction_columns = {
        "news_id", "old_news_type", "new_news_type", "rule_id", "content_sha256",
    }
    if not annotation_columns.issubset(annotations.columns):
        raise ValueError("Annotations are missing correction identity/label fields.")
    if not correction_columns.issubset(corrections.columns):
        raise ValueError("Corrections are missing approved audit fields.")
    for name, frame in (("annotations", annotations), ("corrections", corrections)):
        if frame["news_id"].isna().any() or frame["news_id"].astype(str).str.strip().eq("").any():
            raise ValueError(f"news_id must be nonempty in {name}.")
        if not frame["news_id"].is_unique:
            raise ValueError(f"news_id must be unique in {name}.")
    if not corrections["news_id"].isin(annotations["news_id"]).all():
        raise ValueError("An approved correction references an unknown article.")
    gold_revision = (
        corrections["rule_id"].eq(GOLD_PRICE_LABEL_RULE)
        & corrections["old_news_type"].eq("mention_only")
        & corrections["new_news_type"].eq("industry")
    )
    reviewed_revision = (
        corrections["rule_id"].eq(COMPANY_AWARDS_LISTING_RULE)
        & corrections["news_id"].isin(APPROVED_COMPANY_REVIEW_LABELS)
        & corrections["old_news_type"].eq(corrections["news_id"].map(
            {news_id: labels[0] for news_id, labels in APPROVED_COMPANY_REVIEW_LABELS.items()}
        ))
        & corrections["new_news_type"].eq(corrections["news_id"].map(
            {news_id: labels[1] for news_id, labels in APPROVED_COMPANY_REVIEW_LABELS.items()}
        ))
    )
    if not (gold_revision | reviewed_revision).fillna(False).all():
        raise ValueError("Only explicitly approved label revisions and article-label pairs are supported.")
    reviewed_hashes = (
        corrections["reviewed_content_sha256"].fillna("").astype(str)
        if "reviewed_content_sha256" in corrections
        else pd.Series("", index=corrections.index)
    )
    has_reviewed_hash = reviewed_hashes.ne("")
    allowed_reviewed_hash = (
        reviewed_revision & corrections["news_id"].eq(REVIEWED_CONTENT_ARTICLE_ID)
        & reviewed_hashes.eq(REVIEWED_CONTENT_SHA256)
    )
    if (has_reviewed_hash & ~allowed_reviewed_hash).any():
        raise ValueError("A post-clean content fingerprint is only approved for the reviewed listing article.")
    reviewed_hash_by_id = dict(zip(corrections["news_id"], reviewed_hashes))

    corrected = annotations.copy(deep=True)
    article_indices = dict(zip(annotations["news_id"], annotations.index))
    if not annotations.index.is_unique:
        raise ValueError("Annotation row indices must be unique.")
    for correction in corrections.itertuples(index=False):
        index = article_indices[correction.news_id]
        article = annotations.loc[index]
        if article["annotation_status"] != "labeled":
            raise ValueError(f"Article is not finalized: {correction.news_id}")
        if article["news_type"] not in (correction.old_news_type, correction.new_news_type):
            raise ValueError(f"Article label changed outside the audit: {correction.news_id}")
        content = article["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Article has no readable body: {correction.news_id}")
        fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if fingerprint not in (correction.content_sha256, reviewed_hash_by_id[correction.news_id]):
            raise ValueError(f"Article content changed since review: {correction.news_id}")
        corrected.at[index, "news_type"] = correction.new_news_type
    return corrected


def normalize_news_type(value: object) -> str:
    """Normalize legacy/merged news_type labels into the 7 official categories."""
    if value is None or pd.isna(value):
        return ""
    label = str(value).strip()
    return MERGED_NEWS_TYPE_LABELS.get(label, label)

# Candidates only: these expressions must never be written into news_type.
# Use title/summary rather than content, which can contain unrelated footer news.
PRIORITY_PATTERNS = {
    "macro": [
        r"\b(?:GDP|CPI|Fed)\b", r"lạm phát", r"lãi suất điều hành",
        r"chính sách (?:tiền tệ|tài khóa)", r"tỷ giá",
        r"kinh tế (?:Việt Nam|toàn cầu|thế giới|vĩ mô)",
        r"thuế đối ứng|thuế quan|chiến tranh thương mại",
        r"nghị quyết\s*68", r"tăng trưởng kinh tế",
    ],
    "legal_compliance": [
        r"xử phạt|bị phạt", r"thanh tra|điều tra|khởi tố",
        r"khởi kiện|kiện tụng|tranh chấp|tòa án",
        r"truy thu|vi phạm|cưỡng chế|chậm nộp thuế",
    ],
    "industry": [
        r"toàn ngành|xu hướng ngành|triển vọng ngành",
        r"ngành (?:bán lẻ|dược|thủy sản|sữa|đường|bia|thép|chăn nuôi)",
        r"giá (?:cá tra|tôm|heo|lợn|sữa|đường|vàng|nguyên liệu|thịt)",
        r"cung cầu|chuỗi cung ứng|thị trường xuất khẩu",
        r"thuế chống bán phá giá|quy định.*(?:ngành|xuất khẩu)",
    ],
}

DEFAULT_PRIORITY_QUOTAS = {"macro": 150, "legal_compliance": 100, "industry": 150}


def split_ticker_tokens(value: object) -> list[str]:
    """Return unique ticker tokens while preserving their original order."""
    if value is None or pd.isna(value):
        return []
    tokens = [token.strip().upper() for token in re.split(r"\s*,\s*", str(value))]
    return list(dict.fromkeys(token for token in tokens if token))


def add_sampling_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add metadata used only to construct and audit the annotation sample."""
    required_columns = {"news_id", "source", "content", "published_date", "tickers"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    prepared = data.copy()
    published_dates = pd.to_datetime(prepared["published_date"], errors="coerce")
    if published_dates.isna().any():
        raise ValueError("published_date contains invalid values")

    prepared["sampling_year"] = published_dates.dt.year.astype(int)
    prepared["content_length"] = prepared["content"].fillna("").astype(str).str.len()
    ranked_lengths = prepared["content_length"].rank(method="first")
    prepared["content_length_group"] = pd.qcut(
        ranked_lengths,
        q=4,
        labels=CONTENT_LENGTH_LABELS,
    ).astype(str)
    prepared["_ticker_tokens"] = prepared["tickers"].map(split_ticker_tokens)
    return prepared


def _proportional_group_sample(
    data: pd.DataFrame,
    group_column: str,
    sample_size: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample proportionally by one group while preserving the requested size."""
    group_counts = data[group_column].value_counts(sort=False).sort_index()
    exact_quotas = group_counts / len(data) * sample_size
    quotas = np.floor(exact_quotas).astype(int)

    remaining = sample_size - int(quotas.sum())
    quota_order = (exact_quotas - quotas).sort_values(ascending=False).index
    for group_value in quota_order[:remaining]:
        quotas.loc[group_value] += 1

    sampled_groups = []
    for offset, group_value in enumerate(group_counts.index):
        group = data.loc[data[group_column].eq(group_value)]
        group_size = int(quotas.loc[group_value])
        if group_size:
            sampled_groups.append(
                group.sample(n=group_size, random_state=random_seed + offset)
            )
    return pd.concat(sampled_groups, ignore_index=False)


def _inverse_sqrt_frequency(values: Iterable[object]) -> dict[object, float]:
    counts = pd.Series(list(values)).value_counts()
    return {key: 1.0 / math.sqrt(count) for key, count in counts.items()}


def _calculate_coverage_weights(data: pd.DataFrame) -> pd.Series:
    source_weights = _inverse_sqrt_frequency(data["source"])
    year_weights = _inverse_sqrt_frequency(data["sampling_year"])
    length_weights = _inverse_sqrt_frequency(data["content_length_group"])

    all_tickers = [ticker for tokens in data["_ticker_tokens"] for ticker in tokens]
    ticker_weights = _inverse_sqrt_frequency(all_tickers)

    row_ticker_weights = data["_ticker_tokens"].map(
        lambda tokens: max((ticker_weights[ticker] for ticker in tokens), default=0.0)
    )
    return (
        data["source"].map(source_weights)
        + data["sampling_year"].map(year_weights)
        + data["content_length_group"].map(length_weights)
        + row_ticker_weights
    )


def _select_missing_coverage(
    pool: pd.DataFrame,
    selected: pd.DataFrame,
    random_seed: int,
) -> list[object]:
    """Select rows that cover metadata groups absent from the representative sample."""
    rng = np.random.default_rng(random_seed)
    selected_indices: list[object] = []
    available = pool.copy()

    group_specs = [
        ("source", sorted(set(pool["source"]).union(selected["source"]))),
        ("sampling_year", sorted(set(pool["sampling_year"]).union(selected["sampling_year"]))),
        (
            "content_length_group",
            CONTENT_LENGTH_LABELS,
        ),
    ]

    for column, all_values in group_specs:
        covered = set(selected[column]).union(available.loc[selected_indices, column])
        for value in all_values:
            if value in covered:
                continue
            candidates = available.loc[
                available[column].eq(value) & ~available.index.isin(selected_indices)
            ]
            if not candidates.empty:
                selected_indices.append(rng.choice(candidates.index.to_numpy()))
                covered.add(value)

    all_tickers = sorted({ticker for tokens in pool["_ticker_tokens"] for ticker in tokens})
    covered_tickers = {
        ticker for tokens in selected["_ticker_tokens"] for ticker in tokens
    }
    for ticker in all_tickers:
        if ticker in covered_tickers:
            continue
        candidates = available.loc[
            available["_ticker_tokens"].map(lambda tokens: ticker in tokens)
            & ~available.index.isin(selected_indices)
        ]
        if not candidates.empty:
            chosen_index = rng.choice(candidates.index.to_numpy())
            selected_indices.append(chosen_index)
            covered_tickers.update(available.at[chosen_index, "_ticker_tokens"])

    return selected_indices


def build_annotation_sample(
    data: pd.DataFrame,
    sample_size: int = 400,
    representative_ratio: float = 0.70,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Build a representative sample plus a metadata-coverage supplement."""
    if not 0 < sample_size <= len(data):
        raise ValueError("sample_size must be between 1 and the number of rows")
    if not 0 < representative_ratio <= 1:
        raise ValueError("representative_ratio must be in (0, 1]")

    prepared = add_sampling_features(data).set_index("news_id", drop=False)
    if not prepared.index.is_unique:
        raise ValueError("news_id must be unique before sampling")

    representative_size = round(sample_size * representative_ratio)
    representative = _proportional_group_sample(
        prepared,
        group_column="sampling_year",
        sample_size=representative_size,
        random_seed=random_seed,
    ).copy()
    representative["sampling_partition"] = "representative"

    boost_size = sample_size - len(representative)
    if not boost_size:
        result = representative
    else:
        pool = prepared.drop(index=representative.index).copy()
        required_indices = _select_missing_coverage(
            pool=pool,
            selected=representative,
            random_seed=random_seed + 1,
        )
        if len(required_indices) > boost_size:
            required_indices = required_indices[:boost_size]

        coverage_boost = pool.loc[required_indices].copy()
        remaining_size = boost_size - len(coverage_boost)
        if remaining_size:
            remaining_pool = pool.drop(index=required_indices)
            weights = _calculate_coverage_weights(remaining_pool)
            weighted_rows = remaining_pool.sample(
                n=remaining_size,
                weights=weights,
                random_state=random_seed + 2,
            )
            coverage_boost = pd.concat([coverage_boost, weighted_rows])

        coverage_boost["sampling_partition"] = "coverage_boost"
        result = pd.concat([representative, coverage_boost])

    result = result.sample(frac=1, random_state=random_seed + 3).reset_index(drop=True)
    result["news_type"] = pd.NA
    result["annotation_status"] = pd.NA
    result["annotator_note"] = pd.NA

    output_columns = list(data.columns) + [
        "sampling_partition",
        "sampling_year",
        "content_length_group",
        "news_type",
        "annotation_status",
        "annotator_note",
    ]
    return result[output_columns]


def summarize_sample_coverage(
    full_data: pd.DataFrame,
    sample_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compare metadata coverage between the clean dataset and annotation sample."""
    full_prepared = add_sampling_features(full_data)
    sample_prepared = sample_data.copy()
    sample_prepared["_ticker_tokens"] = sample_prepared["tickers"].map(split_ticker_tokens)
    if "sampling_year" not in sample_prepared:
        sample_prepared["sampling_year"] = pd.to_datetime(
            sample_prepared["published_date"], errors="raise"
        ).dt.year.astype(int)
    if "content_length_group" not in sample_prepared:
        raise ValueError("sample_data must preserve content_length_group from sampling")
    reports = []

    dimensions = {
        "source": "source",
        "year": "sampling_year",
        "content_length_group": "content_length_group",
    }
    for dimension, column in dimensions.items():
        full_counts = full_prepared[column].value_counts()
        sample_counts = sample_prepared[column].value_counts()
        for value in sorted(full_counts.index, key=str):
            reports.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "full_count": int(full_counts[value]),
                    "sample_count": int(sample_counts.get(value, 0)),
                }
            )

    full_tickers = full_prepared["_ticker_tokens"].explode().value_counts()
    sample_tickers = sample_prepared["_ticker_tokens"].explode().value_counts()
    for ticker in sorted(full_tickers.index):
        reports.append(
            {
                "dimension": "ticker",
                "value": ticker,
                "full_count": int(full_tickers[ticker]),
                "sample_count": int(sample_tickers.get(ticker, 0)),
            }
        )

    return pd.DataFrame(reports)


def _check_existing_annotations(
    full_data: pd.DataFrame, existing_annotations: pd.DataFrame,
) -> None:
    required = NEWS_COLUMNS + ANNOTATION_COLUMNS
    missing = set(required).difference(existing_annotations.columns)
    if missing:
        raise ValueError(f"Missing annotation columns: {sorted(missing)}")
    for name, frame in [("clean data", full_data), ("annotations", existing_annotations)]:
        if frame["news_id"].fillna("").eq("").any() or not frame["news_id"].is_unique:
            raise ValueError(f"news_id must be nonempty and unique in {name}")

    original = full_data.set_index("news_id").fillna("")
    annotations = existing_annotations.set_index("news_id").fillna("")
    unknown_ids = annotations.index.difference(original.index)
    if len(unknown_ids):
        raise ValueError(f"Annotations contain {len(unknown_ids)} unknown news_id values")
    columns = NEWS_COLUMNS[1:]
    if annotations[columns].ne(original.loc[annotations.index, columns]).any().any():
        raise ValueError("Original news fields were changed in the annotation file")


def _candidate_scores(pool: pd.DataFrame, patterns: list[str]) -> pd.Series:
    scores = pd.Series(0, index=pool.index, dtype=int)
    for pattern in patterns:
        title_match = pool["title"].fillna("").str.contains(pattern, case=False, regex=True)
        summary_match = pool["summary"].fillna("").str.contains(pattern, case=False, regex=True)
        scores += title_match.astype(int) * 2 + summary_match.astype(int)
    return scores


def extend_annotation_sample(
    full_data: pd.DataFrame,
    existing_annotations: pd.DataFrame,
    target_size: int = 1200,
    random_seed: int = 42,
    priority_quotas: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add unlabelled articles without changing existing human annotations.

    Return new articles, the combined annotation table, selection lineage and
    candidate availability. Priority groups describe a sampling intention, not a
    predicted or confirmed label. Unfilled quotas become random sample slots.
    """
    _check_existing_annotations(full_data, existing_annotations)
    if not len(existing_annotations) < target_size <= len(full_data):
        raise ValueError("target_size must exceed current size and fit in clean data")
    quotas = DEFAULT_PRIORITY_QUOTAS if priority_quotas is None else priority_quotas
    if any(group not in PRIORITY_PATTERNS or quota < 0 for group, quota in quotas.items()):
        raise ValueError("Priority quotas must be nonnegative and use known groups")
    additional_size = target_size - len(existing_annotations)
    if sum(quotas.values()) > additional_size:
        raise ValueError("Priority quotas exceed the additional sample size")

    ordered_data = full_data[NEWS_COLUMNS].sort_values("news_id").reset_index(drop=True)
    prepared = add_sampling_features(ordered_data).set_index("news_id", drop=False)
    pool = prepared.drop(index=existing_annotations["news_id"]).copy()
    selected_groups = []
    availability = []

    for offset, (group, quota) in enumerate(quotas.items()):
        scores = _candidate_scores(pool, PRIORITY_PATTERNS[group])
        candidates = pool.loc[scores.gt(0)]
        take = min(quota, len(candidates))
        chosen = candidates.sample(
            n=take, weights=scores.loc[candidates.index] if take else None,
            random_state=random_seed + offset,
        ).copy()
        chosen["selection_method"] = "keyword_candidates"
        chosen["priority_group"] = group
        chosen["keyword_score"] = scores.loc[chosen.index]
        selected_groups.append(chosen)
        availability.append({
            "priority_group": group, "requested_candidates": quota,
            "available_at_selection": len(candidates), "selected_count": take,
        })
        pool = pool.drop(index=chosen.index)

    random_size = additional_size - sum(len(group) for group in selected_groups)
    if random_size:
        random_rows = _proportional_group_sample(
            pool, "sampling_year", random_size, random_seed + 100,
        ).copy()
        random_rows["selection_method"] = "year_stratified_random"
        random_rows["priority_group"] = ""
        random_rows["keyword_score"] = 0
        selected_groups.append(random_rows)

    selected = pd.concat(selected_groups).sample(
        frac=1, random_state=random_seed + 200,
    ).reset_index(drop=True)
    new_articles = selected[NEWS_COLUMNS].copy()
    for column in ANNOTATION_COLUMNS:
        new_articles[column] = ""
    existing = existing_annotations[NEWS_COLUMNS + ANNOTATION_COLUMNS].fillna("")
    combined = pd.concat([existing, new_articles], ignore_index=True)

    lineage_columns = ["news_id", "selection_method", "priority_group", "keyword_score"]
    lineage = selected[lineage_columns].copy()
    lineage["random_seed"] = random_seed
    lineage["sampling_year"] = selected["sampling_year"]
    lineage["content_length_group"] = selected["content_length_group"]
    return new_articles, combined, lineage, pd.DataFrame(availability)


def save_annotation_csv(data: pd.DataFrame, path: Path) -> None:
    """Export annotation data, refusing to erase any work saved in a prior run."""
    if path.exists() and path.stat().st_size > 3:
        saved = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if not set(["news_id"] + ANNOTATION_COLUMNS).issubset(saved.columns):
            raise ValueError(f"Existing file has an unexpected schema: {path.name}")
        if not saved["news_id"].is_unique:
            raise ValueError(f"Existing file has duplicate news_id: {path.name}")
        annotated = saved[ANNOTATION_COLUMNS].apply(
            lambda column: column.str.strip().ne("")
        ).any(axis=1)
        protected = saved.loc[annotated].set_index("news_id")[ANNOTATION_COLUMNS]
        proposed = data.fillna("").set_index("news_id")[ANNOTATION_COLUMNS]
        if not protected.index.isin(proposed.index).all():
            raise FileExistsError(f"Export would remove annotations in {path.name}")
        if not protected.eq(proposed.reindex(protected.index)).all().all():
            raise FileExistsError(f"Export would overwrite annotations in {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")


def refresh_annotation_content(
    annotations: pd.DataFrame,
    clean_news: pd.DataFrame,
    excluded_records: pd.DataFrame,
    duplicate_merge_log: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh content only, preserving every human field and excluded article ID.

    The returned audit is an in-memory training eligibility check, not new labels.
    Excluded articles remain in the annotation file for later recovery/review.
    """
    required = set(NEWS_COLUMNS + ANNOTATION_COLUMNS)
    if not required.issubset(annotations.columns):
        raise ValueError("Annotations are missing required news/annotation fields")
    for name, frame in [("annotations", annotations), ("clean news", clean_news)]:
        if frame["news_id"].eq("").any() or not frame["news_id"].is_unique:
            raise ValueError(f"news_id must be nonempty and unique in {name}")

    refreshed = annotations.copy()
    clean_by_id = clean_news.set_index("news_id")
    clean_ids_by_url = clean_news.set_index("url")["news_id"].to_dict()
    excluded_by_id = excluded_records.drop_duplicates("news_id").set_index("news_id")
    canonical_ids = {news_id: news_id for news_id in clean_by_id.index}
    for row in duplicate_merge_log.itertuples(index=False):
        for news_id in str(row.merged_news_ids).split("|"):
            canonical_ids[news_id] = row.kept_news_id

    audit_rows = []
    for index, row in annotations.iterrows():
        news_id = row["news_id"]
        canonical_id = canonical_ids.get(news_id, "")
        if canonical_id not in clean_by_id.index:
            # A bad duplicate may have been excluded before consolidation while
            # another crawl of the exact same canonical URL has a usable body.
            canonical_id = clean_ids_by_url.get(row["url"], "")
        if canonical_id in clean_by_id.index:
            content = clean_by_id.at[canonical_id, "content"]
            issue = ""
        elif news_id in excluded_by_id.index:
            excluded = excluded_by_id.loc[news_id]
            content = excluded.get("content", "")
            issue = str(excluded["exclusion_reason"])
            if issue.startswith("CONTENT_QUALITY:"):
                # The excluded log retains the diagnostic text. A related-news
                # list or mismatched article must not masquerade as clean content.
                content = ""
        else:
            raise ValueError(f"Annotation ID has no cleaning lineage: {news_id}")
        if pd.isna(content):
            content = ""
        refreshed.at[index, "content"] = str(content)
        normalized_label = normalize_news_type(row["news_type"])
        audit_rows.append({
            "news_id": news_id,
            "canonical_news_id": canonical_id,
            "training_news_type": normalized_label,
            "content_changed": str(content) != str(row["content"]),
            "content_issue": issue,
            "training_eligible": (
                not issue and row["annotation_status"] == "labeled"
                and bool(normalized_label)
            ),
        })

    audit = pd.DataFrame(audit_rows, index=annotations.index)
    # Two labels for a newly merged article must not enter a model as independent
    # observations. A conflicting pair is quarantined without changing either label.
    eligible = audit.loc[audit["training_eligible"]]
    for canonical_id, group in eligible.groupby("canonical_news_id", sort=False):
        if len(group) < 2:
            continue
        indices = group.index
        if audit.loc[indices, "training_news_type"].nunique() > 1:
            audit.loc[indices, "training_eligible"] = False
            audit.loc[indices, "content_issue"] = "DUPLICATE_LABEL_CONFLICT"
        else:
            preferred = group.loc[group["news_id"].eq(canonical_id)]
            kept_index = preferred.index[0] if len(preferred) else indices[0]
            discarded = indices.drop(kept_index)
            audit.loc[discarded, "training_eligible"] = False
            audit.loc[discarded, "content_issue"] = "DUPLICATE_ARTICLE"

    protected_columns = annotations.columns.difference(["content"], sort=False)
    if not refreshed[protected_columns].equals(annotations[protected_columns]):
        raise AssertionError("Refreshing content changed protected annotation fields")
    return refreshed, audit
