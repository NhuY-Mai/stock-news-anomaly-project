"""Select one annotation supplement with auditable heuristics and source quotas."""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits

from .annotation_utils import (
    ANNOTATION_COLUMNS,
    NEWS_COLUMNS,
    PRIORITY_PATTERNS,
    _candidate_scores,
    _check_existing_annotations,
    split_ticker_tokens,
)


TOPIC_PATTERNS = {
    **PRIORITY_PATTERNS,
    "financial_results": [
        r"doanh thu|lợi nhuận|báo cáo tài chính|kết quả kinh doanh|\bKQKD\b",
        r"biên lợi nhuận|dòng tiền|báo lãi|báo lỗ|kế hoạch kinh doanh",
    ],
    "governance_capital": [
        r"cổ tức|phát hành|\bESOP\b|cổ phiếu quỹ|cổ đông lớn",
        r"giao dịch nội bộ|đăng ký (?:mua|bán)|hội đồng quản trị|bổ nhiệm|miễn nhiệm",
    ],
    "business_operations": [
        r"mở rộng|nhà máy|dự án|công suất|sản phẩm mới|hợp đồng",
        r"mua lại|sáp nhập|thoái vốn|liên doanh|phát triển bền vững|\bESG\b",
    ],
    "stock_market": [
        r"VN.?Index|giá cổ phiếu|thanh khoản|khối ngoại|giá mục tiêu",
        r"khuyến nghị|định giá|phiên giao dịch|tăng trần|giảm sàn",
    ],
}


def allocate_source_quotas(
    counts: pd.Series, sample_size: int, max_share: float = 0.1875,
) -> pd.Series:
    """Allocate by square-root frequency, respecting available rows and capacity."""
    counts = counts[counts.gt(0)].sort_index().astype(int)
    if not 0 < sample_size <= int(counts.sum()):
        raise ValueError("sample_size must fit available source counts")
    if not 0 < max_share <= 1:
        raise ValueError("max_share must be in (0, 1]")

    cap = max(1, math.ceil(sample_size * max_share))
    while int(counts.clip(upper=cap).sum()) < sample_size:
        cap += 1
    capacities = counts.clip(upper=cap)
    quotas = pd.Series(int(sample_size >= len(counts)), index=counts.index)
    weights = np.sqrt(counts.to_numpy(dtype=float))
    for _ in range(sample_size - int(quotas.sum())):
        priorities = weights / (quotas.to_numpy() + 1)
        priorities[quotas.to_numpy() >= capacities.to_numpy()] = -np.inf
        quotas.iloc[int(np.argmax(priorities))] += 1
    return quotas.astype(int)


def _text_regions(
    ordered_data: pd.DataFrame, existing: pd.DataFrame, random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure lexical regions, proximity to pending reviews and text novelty."""
    texts = (
        ordered_data["title"].fillna("") + " "
        + ordered_data["summary"].fillna("")
    )
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_features=16000,
        sublinear_tf=True, dtype=np.float32,
    )
    vectors = vectorizer.fit_transform(texts)
    unique_vectors = len({
        (tuple(vectors[i].indices), tuple(vectors[i].data.round(6)))
        for i in range(len(ordered_data))
    })
    cluster_count = min(40, max(1, len(ordered_data) // 10), unique_vectors)
    indices = pd.Index(ordered_data["news_id"])

    with threadpool_limits(limits=1):
        if cluster_count > 1:
            clusters = MiniBatchKMeans(
                n_clusters=cluster_count, random_state=random_seed,
                n_init=3, batch_size=256, max_iter=100,
            ).fit_predict(vectors)
        else:
            clusters = np.zeros(len(ordered_data), dtype=int)

        def nearest_similarity(status: str) -> np.ndarray:
            reference_ids = existing.loc[
                existing["annotation_status"].eq(status), "news_id"
            ]
            positions = indices.get_indexer(reference_ids)
            if not len(positions):
                return np.zeros(len(ordered_data))
            return cosine_similarity(vectors, vectors[positions]).max(axis=1)

        review_similarity = nearest_similarity("review_required")
        novelty = 1 - nearest_similarity("labeled")
    return clusters, review_similarity, novelty


def _sampling_features(
    full_data: pd.DataFrame, existing: pd.DataFrame, random_seed: int,
) -> pd.DataFrame:
    ordered = full_data[NEWS_COLUMNS].sort_values("news_id").reset_index(drop=True)
    features = ordered.copy()
    clusters, review_similarity, novelty = _text_regions(ordered, existing, random_seed)
    features["topic_cluster"] = clusters
    features["review_similarity"] = review_similarity
    features["novelty_score"] = np.clip(novelty, 0, 1)
    features["quarter"] = pd.to_datetime(features["published_date"]).dt.to_period("Q").astype(str)
    features["publisher_group"] = features["source"].map(_publisher_group)

    topic_scores = pd.DataFrame({
        label: _candidate_scores(features, patterns)
        for label, patterns in TOPIC_PATTERNS.items()
    })
    confirmed = existing.loc[existing["annotation_status"].eq("labeled"), "news_type"]
    label_counts = confirmed.value_counts()
    threshold = float(label_counts.median()) / 2 if len(label_counts) else 1
    rare_labels = [label for label in TOPIC_PATTERNS if label_counts.get(label, 0) <= threshold]
    rare_scores = pd.DataFrame({
        label: topic_scores[label].clip(upper=6) / math.sqrt(1 + label_counts.get(label, 0))
        for label in rare_labels
    })
    features["rare_label_score"] = rare_scores.max(axis=1) if rare_labels else 0.0
    features["priority_hints"] = [
        "|".join(label for label in rare_labels if topic_scores.at[i, label] > 0)
        for i in features.index
    ]

    features["topic_signal_count"] = topic_scores.gt(0).sum(axis=1)
    overlap = ((features["topic_signal_count"] - 1) / 3).clip(0, 1)
    near_review = ((features["review_similarity"] - 0.15) / 0.45).clip(0, 1)
    features["ambiguity_score"] = 0.6 * overlap + 0.4 * near_review
    features["ambiguous_candidate"] = (
        features["topic_signal_count"].ge(2) | features["review_similarity"].ge(0.30)
    )

    content = features["content"].fillna("")
    lengths = content.str.len()
    lower_bound, upper_bound = lengths.quantile([0.02, 0.95])
    flags = pd.DataFrame({
        "very_short": lengths.le(lower_bound),
        "very_long": lengths.ge(upper_bound),
        "missing_summary": features["summary"].fillna("").str.strip().eq(""),
        "possible_footer": content.str.contains(
            r"Bài liên quan|Cùng chuyên mục|Xem thêm bài viết tại|reCAPTCHA|Đăng nhập Đăng ký",
            case=False, regex=True,
        ),
        "replacement_character": content.str.contains("\ufffd", regex=False),
    })
    features["anomaly_flags"] = [
        "|".join(column for column in flags if flags.at[i, column])
        for i in features.index
    ]
    features["anomaly_score"] = (flags.astype(float).sum(axis=1) / 2).clip(0, 1)
    features["content_length"] = lengths
    features["short_length_boundary"] = lower_bound
    features["long_length_boundary"] = upper_bound
    return features.loc[~features["news_id"].isin(existing["news_id"])].reset_index(drop=True)


def _publisher_group(source: str) -> str:
    """Keep source values intact but combine known feeds from one publisher."""
    name = re.sub(r"^(?:API|RSS):\s*", "", source, flags=re.IGNORECASE)
    if name.lower().startswith("cafef"):
        return "CafeF"
    if name in {"Thanh Niên", "ThanhNien"}:
        return "ThanhNien"
    return name


def _publisher_aware_source_quotas(pool: pd.DataFrame, sample_size: int) -> pd.Series:
    publisher_counts = pool["publisher_group"].value_counts()
    publisher_quotas = allocate_source_quotas(publisher_counts, sample_size)
    quotas = []
    for publisher, quota in publisher_quotas.items():
        counts = pool.loc[pool["publisher_group"].eq(publisher), "source"].value_counts()
        if quota:
            quotas.append(allocate_source_quotas(counts, int(quota), max_share=1.0))
        else:
            quotas.append(pd.Series(0, index=counts.index, dtype=int))
    return pd.concat(quotas).sort_index()


class _DiverseSelector:
    """Greedy acquisition with exact source quotas and soft coverage targets."""

    def __init__(self, pool: pd.DataFrame, quotas: pd.Series, random_seed: int):
        self.pool = pool
        self.quotas = quotas
        self.source_codes = pd.Categorical(pool["source"], categories=quotas.index).codes
        self.source_used = np.zeros(len(quotas), dtype=int)
        self.selected = np.zeros(len(pool), dtype=bool)
        self.tie_breaker = np.random.default_rng(random_seed).uniform(0, 1e-8, len(pool))
        self.coverage = []

        ticker_tokens = pool["tickers"].map(split_ticker_tokens)
        all_tickers = sorted({ticker for tokens in ticker_tokens for ticker in tokens})
        ticker_matrix = np.array([[ticker in tokens for ticker in all_tickers] for tokens in ticker_tokens])
        self._add_coverage(ticker_matrix, float(quotas.sum()) * ticker_matrix.sum(axis=1).mean())
        for column in ["quarter", "topic_cluster"]:
            matrix = pd.get_dummies(pool[column]).to_numpy(dtype=bool)
            self._add_coverage(matrix, float(quotas.sum()))

    def _add_coverage(self, matrix: np.ndarray, budget: float) -> None:
        weights = np.sqrt(matrix.sum(axis=0).astype(float))
        targets = budget * weights / weights.sum()
        self.coverage.append((matrix, targets, np.zeros(len(targets))))

    def take(
        self, eligible: np.ndarray, count: int, base_scores: np.ndarray, reason: str,
    ) -> list[tuple[int, str]]:
        result = []
        positive_scores = base_scores[eligible]
        maximum = positive_scores.max() if len(positive_scores) else 1
        normalized = base_scores / maximum if maximum > 0 else base_scores
        for _ in range(count):
            allowed = (
                eligible & ~self.selected
                & (self.source_used[self.source_codes] < self.quotas.to_numpy()[self.source_codes])
            )
            if not allowed.any():
                break

            score = normalized.copy()
            for weight, (matrix, targets, used) in zip([0.65, 0.20, 0.30], self.coverage):
                deficits = np.clip((targets - used) / targets, 0, 1)
                boost = (matrix * deficits).sum(axis=1) / matrix.sum(axis=1).clip(min=1)
                score += weight * boost
            remaining_source = (
                self.quotas.to_numpy() - self.source_used
            ) / np.maximum(self.quotas.to_numpy(), 1)
            score += 0.20 * remaining_source[self.source_codes] + self.tie_breaker
            score[~allowed] = -np.inf
            chosen = int(np.argmax(score))
            self.selected[chosen] = True
            self.source_used[self.source_codes[chosen]] += 1
            for matrix, _, used in self.coverage:
                used += matrix[chosen]
            result.append((chosen, reason))
        return result


def build_strategic_annotation_sample(
    full_data: pd.DataFrame,
    existing_annotations: pd.DataFrame,
    sample_size: int = 800,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select one supplement; acquisition scores and hints never become labels.

    Fractions reserve up to 37.5% for rare-label candidates, 31.25% for ambiguous
    cases, and 12.5% for unusual text. Unfilled slots use diversity/novelty.
    Source quotas are exact; ticker, quarter and lexical-region targets are soft.
    """
    _check_existing_annotations(full_data, existing_annotations)
    available_count = len(full_data) - len(existing_annotations)
    if not 0 < sample_size <= available_count:
        raise ValueError("sample_size must fit articles outside the existing sample")
    pool = _sampling_features(full_data, existing_annotations, random_seed)
    quotas = _publisher_aware_source_quotas(pool, sample_size)
    selector = _DiverseSelector(pool, quotas, random_seed)
    selected_rows = []
    stages = [
        ("rare_label_candidates", "rare_label_score", pool["rare_label_score"].gt(0), 0.375),
        ("ambiguous_or_review_neighbors", "ambiguity_score", pool["ambiguous_candidate"], 0.3125),
        ("text_anomalies", "anomaly_score", pool["anomaly_score"].gt(0), 0.125),
    ]
    for reason, score_column, eligible, fraction in stages:
        selected_rows.extend(selector.take(
            eligible.to_numpy(), int(sample_size * fraction),
            pool[score_column].to_numpy(), reason,
        ))
    selected_rows.extend(selector.take(
        np.ones(len(pool), dtype=bool), sample_size - len(selected_rows),
        pool["novelty_score"].to_numpy(), "coverage_and_novelty",
    ))
    positions = [position for position, _ in selected_rows]
    if len(positions) != sample_size or len(set(positions)) != sample_size:
        raise RuntimeError("Selection did not satisfy the requested sample size")

    log = pool.iloc[positions].copy().reset_index(drop=True)
    log["selection_reason"] = [reason for _, reason in selected_rows]
    log["random_seed"] = random_seed
    log = log.drop(columns=["title", "summary", "content"])
    additional = pool.iloc[positions][NEWS_COLUMNS].copy()
    for column in ANNOTATION_COLUMNS:
        additional[column] = ""
    additional = additional.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    existing = existing_annotations[NEWS_COLUMNS + ANNOTATION_COLUMNS].fillna("")
    combined = pd.concat([existing, additional], ignore_index=True)

    source_report = pd.DataFrame({
        "available_count": pool["source"].value_counts(), "target_count": quotas,
        "selected_count": additional["source"].value_counts(),
    }).fillna(0).astype(int).sort_index().rename_axis("source").reset_index()
    if not source_report["target_count"].equals(source_report["selected_count"]):
        raise RuntimeError("Source quotas were not achieved exactly")
    return additional, combined, log, source_report


def summarize_strategic_coverage(
    full_data: pd.DataFrame,
    existing: pd.DataFrame,
    additional: pd.DataFrame,
) -> pd.DataFrame:
    """Report actual source/ticker/time coverage, not assumed label frequencies."""
    dimensions = {}
    for dimension in ["source", "publisher", "ticker", "quarter"]:
        counts = []
        for frame in [full_data, existing, additional]:
            if dimension == "ticker":
                values = frame["tickers"].map(split_ticker_tokens).explode()
            elif dimension == "quarter":
                values = pd.to_datetime(frame["published_date"]).dt.to_period("Q").astype(str)
            elif dimension == "publisher":
                values = frame["source"].map(_publisher_group)
            else:
                values = frame["source"]
            counts.append(values.value_counts())
        report = pd.DataFrame({
            "full_count": counts[0], "existing_count": counts[1],
            "selected_count": counts[2],
        }).fillna(0).astype(int)
        report["combined_count"] = report["existing_count"] + report["selected_count"]
        report["dimension"] = dimension
        report.index.name = "value"
        dimensions[dimension] = report.reset_index()
    return pd.concat(dimensions.values(), ignore_index=True)
