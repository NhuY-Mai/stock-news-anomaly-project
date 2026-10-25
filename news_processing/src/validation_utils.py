"""Small, reusable helpers for raw-news validation notebooks."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import pandas as pd


VIETNAM_TIMEZONE = "Asia/Ho_Chi_Minh"


def canonicalize_url(value: object) -> str:
    """Remove mobile/www prefixes, query strings, fragments, and trailing slashes."""
    if pd.isna(value):
        return ""

    parts = urlsplit(str(value).strip())
    hostname = re.sub(r"^(www\.|m\.)", "", parts.netloc.lower())
    path = re.sub(r"/+$", "", parts.path) or "/"
    return urlunsplit((parts.scheme.lower(), hostname, path, "", ""))


def is_valid_http_url(value: object) -> bool:
    """Return True only for a scalar HTTP(S) URL with a hostname."""
    if pd.isna(value) or not str(value).strip():
        return False

    parts = urlsplit(str(value).strip())
    return parts.scheme.lower() in {"http", "https"} and bool(parts.netloc)


def summarize_duplicates(series: pd.Series, check_name: str) -> dict[str, int | str]:
    """Summarize rows, groups, and extra rows involved in exact duplicates."""
    duplicate_mask = series.duplicated(keep=False) & series.ne("")
    group_sizes = series[duplicate_mask].value_counts()
    return {
        "check": check_name,
        "rows_in_duplicate_groups": int(duplicate_mask.sum()),
        "duplicate_groups": int(len(group_sizes)),
        "extra_rows": int((group_sizes - 1).sum()),
    }


def normalize_title(value: object) -> str:
    """Normalize Unicode, case, and whitespace for exact-title comparison."""
    normalized = unicodedata.normalize("NFKC", str(value)).lower().strip()
    return re.sub(r"\s+", " ", normalized)


def classify_timestamp_format(value: object) -> str:
    """Classify timestamp text before applying an explicit parsing rule."""
    if pd.isna(value) or not str(value).strip():
        return "missing"

    text = str(value).strip()
    if re.fullmatch(r"\d{8}T\d{6}Z", text):
        return "compact_utc"
    if re.match(r"^\d{4}-\d{2}-\d{2}T", text):
        has_timezone = bool(re.search(r"(Z|[+-]\d{2}:?\d{2})$", text))
        return "iso_with_timezone" if has_timezone else "iso_local"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?", text):
        return "iso_local"
    if re.fullmatch(r"\d{1,2}:\d{2}\s+\d{1,2}/\d{1,2}/\d{4}", text):
        return "time_dmy"
    if re.search(r"\b(?:AM|PM)\b", text, flags=re.IGNORECASE):
        return "slash_ampm"
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
        return "slash_date"
    return "unknown"


def parse_timestamp(value: object) -> pd.Timestamp:
    """Parse known source formats and return a timezone-aware Vietnam timestamp."""
    timestamp_format = classify_timestamp_format(value)
    if timestamp_format == "missing" or timestamp_format == "unknown":
        return pd.NaT

    text = str(value).strip()
    if timestamp_format == "compact_utc":
        timestamp = pd.to_datetime(text, format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True)
    elif timestamp_format == "iso_with_timezone":
        timestamp = pd.to_datetime(text, errors="coerce", utc=True)
    elif timestamp_format == "time_dmy":
        timestamp = pd.to_datetime(text, format="%H:%M %d/%m/%Y", errors="coerce")
    elif timestamp_format == "slash_ampm":
        timestamp = pd.to_datetime(text, errors="coerce", dayfirst=False)
    elif timestamp_format == "slash_date":
        timestamp = pd.to_datetime(text, errors="coerce", dayfirst=True)
    else:
        timestamp = pd.to_datetime(text, errors="coerce")

    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(VIETNAM_TIMEZONE)
    return timestamp.tz_convert(VIETNAM_TIMEZONE)


def split_tickers(value: object) -> list[str]:
    """Split a comma-separated ticker field into normalized tokens."""
    return [token.strip().upper() for token in str(value).split(",") if token.strip()]


def build_issue_rows(
    data: pd.DataFrame,
    mask: pd.Series,
    rule_code: str,
    severity: str,
    column: str,
    reason: str,
) -> pd.DataFrame:
    """Build actionable issue rows while limiting long values in the report."""
    selected = data.loc[mask, ["news_id", "source", "url"]].copy()
    selected.insert(0, "row_index", selected.index)
    if selected.empty:
        return selected

    current_values = data.loc[mask, column].astype("string").fillna("<NA>").str.slice(0, 300)
    selected["rule_code"] = rule_code
    selected["severity"] = severity
    selected["column"] = column
    selected["current_value"] = current_values.to_numpy()
    selected["reason"] = reason
    return selected
