"""Rebuild ``train_v2.csv`` = baseline ``train.csv`` + 26 verified review38 articles.

The V2 experiment is frozen, so this script is a *reproduction* tool, not a
generator for new data versions:

* every input (train/validation/test, review38 resolution, clean, mapping,
  duplicate log) must match its locked SHA-256 before anything is built;
* every review38 ``TRAIN_V2`` row must be labeled, map to a known label, exist
  in ``news_clean.csv`` with the exact reviewed content hash, and must not leak
  into validation/test by ``news_id``, ``canonical_news_id``, URL or
  whitespace-normalized ``input_text``;
* the built bytes must equal the locked ``train_v2.csv`` hash;
* the output is written atomically, an identical rerun is a no-op, and a
  different existing file is never overwritten.

Any intended change to the training data must go to a new file/run name
(for example ``train_v3.csv``) instead of editing these locks.

Usage::

    python scripts/build_train_v2.py           # rebuild or confirm no-op
    python scripts/build_train_v2.py --check   # verify only, never writes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.finalize_news_type_review import (  # noqa: E402
    EXPECTED_RESOLUTION_SHA256,
    EXPECTED_V2_ACTION_COUNTS,
    _atomic_write_if_changed,
    _csv_bytes,
    _sha256,
    _sha256_bytes,
)
from src.model_data_utils import MODEL_SPLIT_COLUMNS, build_phobert_input_text  # noqa: E402

SPLIT_COLUMNS = [*MODEL_SPLIT_COLUMNS, "split"]
TRAIN_ACTION = "TRAIN_V2"
EXPECTED_BASELINE_ROWS = 740
EXPECTED_ADDED_ROWS = EXPECTED_V2_ACTION_COUNTS[TRAIN_ACTION]
EXPECTED_OUTPUT_ROWS = EXPECTED_BASELINE_ROWS + EXPECTED_ADDED_ROWS

RELATIVE_PATHS = {
    "train": "data/splits/news_type/train.csv",
    "validation": "data/splits/news_type/validation.csv",
    "test": "data/splits/news_type/test.csv",
    "resolution": "data/annotation/news_type/review38/review38_resolution.csv",
    "clean": "data/processed/news_clean.csv",
    "mapping": "models/news_type_phobert_cnn/label_mapping.json",
    "duplicates": "data/interim/duplicate_merge_log.csv",
    "output": "data/splits/news_type/train_v2.csv",
}

# Locked hashes of the frozen V2 experiment (see reports/news_type/report_final_news_type.md §7
# and reports/news_type/v2/runs/review38_incremental_v2/test_evaluation_config.json).
EXPECTED_SHA256: dict[str, str] = {
    "train": "1121c3d79dba52e32b361f91d3a38713817e2579ad4d3c48bf45c458b92e8b23",
    "validation": "9be546e6cc1bd5b86e35b9fb70e62fdacfc7d2589e04af9001a2b798862eff45",
    "test": "1e40f86ac0e85772ccbea65f0ee815eb78277ec91e0834de91d0c66aa4270bf0",
    "resolution": EXPECTED_RESOLUTION_SHA256,
    "mapping": "cbf2d86bff834bd7762f6a9f98a4038cb1a26b722f464cc48d656a3a6bddb3da",
    "output": "a6dc1caf521f29b28b41fc9aeb71ada84d0c02d17bd4c32efd9f7cc52cf6571f",
}

_WHITESPACE_RE = re.compile(r"\s+")


class TrainV2BuildError(RuntimeError):
    """Raised when inputs or the rebuilt file violate a locked invariant."""


def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(value)).strip()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _verify_input_hashes(paths: Mapping[str, Path], expected: Mapping[str, str]) -> None:
    for name, path in paths.items():
        if name == "output" or name not in expected:
            continue
        if not path.is_file():
            raise TrainV2BuildError(f"Thiếu đầu vào: {path}")
        actual = _sha256(path)
        if actual != expected[name]:
            raise TrainV2BuildError(
                f"SHA-256 của {name} không khớp bản đã khóa: "
                f"expected {expected[name]}, received {actual}"
            )


def _alias_lineage(duplicate_log: pd.DataFrame) -> dict[str, str]:
    lineage: dict[str, str] = {}
    for group in duplicate_log.itertuples(index=False):
        for alias in str(group.merged_news_ids).split("|"):
            if alias:
                lineage[alias] = group.kept_news_id
    return lineage


def _check_split_frame(name: str, frame: pd.DataFrame) -> None:
    if frame.columns.tolist() != SPLIT_COLUMNS:
        raise TrainV2BuildError(f"Cột của {name} khác schema split: {frame.columns.tolist()}")
    for key in ("news_id", "canonical_news_id", "url"):
        duplicated = frame.loc[frame[key].duplicated(), key].tolist()
        if duplicated:
            raise TrainV2BuildError(f"{name} có {key} trùng: {duplicated[:5]}")


def _select_train_rows(resolution: pd.DataFrame, label2id: Mapping[str, int]) -> pd.DataFrame:
    action_counts = resolution["v2_action"].value_counts().to_dict()
    if action_counts != EXPECTED_V2_ACTION_COUNTS:
        raise TrainV2BuildError(
            f"Phân bổ v2_action khác bản đã khóa: {action_counts} != {EXPECTED_V2_ACTION_COUNTS}"
        )
    selected = resolution[resolution["v2_action"] == TRAIN_ACTION]
    if selected["news_id"].duplicated().any():
        raise TrainV2BuildError("review38 có news_id TRAIN_V2 bị lặp.")
    not_labeled = selected.loc[selected["final_annotation_status"] != "labeled", "news_id"]
    if not not_labeled.empty:
        raise TrainV2BuildError(f"Bài TRAIN_V2 chưa ở trạng thái labeled: {not_labeled.tolist()}")
    unknown = sorted(set(selected["final_news_type"]) - set(label2id))
    if unknown:
        raise TrainV2BuildError(f"Nhãn TRAIN_V2 không có trong label mapping: {unknown}")
    return selected


def _build_new_rows(
    selected: pd.DataFrame,
    clean: pd.DataFrame,
    label2id: Mapping[str, int],
    lineage: Mapping[str, str],
) -> pd.DataFrame:
    if clean["news_id"].duplicated().any():
        raise TrainV2BuildError("news_clean.csv có news_id trùng.")
    clean_map = clean.set_index("news_id")
    missing = [news_id for news_id in selected["news_id"] if news_id not in clean_map.index]
    if missing:
        raise TrainV2BuildError(f"news_id TRAIN_V2 không có trong news_clean.csv: {missing}")

    rows = []
    for review in selected.itertuples(index=False):
        article = clean_map.loc[review.news_id]
        content_hash = hashlib.sha256(article["content"].encode("utf-8")).hexdigest()
        if content_hash != review.reviewed_content_sha256:
            raise TrainV2BuildError(
                f"Nội dung clean của {review.news_id} khác nội dung đã review "
                f"(reviewed_content_sha256)."
            )
        rows.append({
            "news_id": review.news_id,
            "canonical_news_id": lineage.get(review.news_id, review.news_id),
            "source": article["source"],
            "url": article["url"],
            "published_date": article["published_date"],
            "tickers": article["tickers"],
            "title": article["title"],
            "summary": article["summary"],
            "content": article["content"],
            "input_text": build_phobert_input_text(
                article["title"], article["summary"], article["content"]
            ),
            "news_type": review.final_news_type,
            "label_id": str(label2id[review.final_news_type]),
            "split": "train",
        })
    return pd.DataFrame(rows, columns=SPLIT_COLUMNS)


def _check_no_leakage(new_rows: pd.DataFrame, others: Mapping[str, pd.DataFrame]) -> None:
    """New rows must not collide with baseline train or held-out splits."""
    keys = {
        "news_id": lambda frame: set(frame["news_id"]),
        "canonical_news_id": lambda frame: set(frame["canonical_news_id"]),
        "url": lambda frame: set(frame["url"]),
        "input_text": lambda frame: set(frame["input_text"].map(_normalize_text)),
    }
    for key, extract in keys.items():
        new_values = extract(new_rows)
        if len(new_values) != len(new_rows):
            raise TrainV2BuildError(f"Các dòng TRAIN_V2 trùng nhau theo {key}.")
        for name, frame in others.items():
            overlap = new_values & extract(frame)
            if overlap:
                raise TrainV2BuildError(
                    f"Rò rỉ/trùng giữa TRAIN_V2 và {name} theo {key}: {sorted(overlap)[:5]}"
                )


def build_train_v2_bytes(
    project_dir: Path = PROJECT_DIR,
    expected_sha256: Mapping[str, str] | None = EXPECTED_SHA256,
) -> tuple[bytes, pd.DataFrame, pd.DataFrame]:
    """Return ``(csv_bytes, baseline_train, new_rows)`` after all checks pass.

    ``expected_sha256=None`` disables hash locks (tests with synthetic data only).
    """
    paths = {name: project_dir / relative for name, relative in RELATIVE_PATHS.items()}
    if expected_sha256 is not None:
        _verify_input_hashes(paths, expected_sha256)

    train = _read_csv(paths["train"])
    validation = _read_csv(paths["validation"])
    test = _read_csv(paths["test"])
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        _check_split_frame(name, frame)
    if expected_sha256 is not None and len(train) != EXPECTED_BASELINE_ROWS:
        raise TrainV2BuildError(f"train.csv có {len(train)} dòng, kỳ vọng {EXPECTED_BASELINE_ROWS}.")

    label_mapping = json.loads(paths["mapping"].read_text(encoding="utf-8-sig"))
    label2id = label_mapping["label2id"]
    lineage = _alias_lineage(_read_csv(paths["duplicates"]))

    selected = _select_train_rows(_read_csv(paths["resolution"]), label2id)
    new_rows = _build_new_rows(selected, _read_csv(paths["clean"]), label2id, lineage)
    _check_no_leakage(new_rows, {"train": train, "validation": validation, "test": test})

    combined = pd.concat([train, new_rows], ignore_index=True)
    payload = _csv_bytes(combined)
    if expected_sha256 is not None:
        if len(combined) != EXPECTED_OUTPUT_ROWS:
            raise TrainV2BuildError(f"train_v2 có {len(combined)} dòng, kỳ vọng {EXPECTED_OUTPUT_ROWS}.")
        actual = _sha256_bytes(payload)
        if actual != expected_sha256["output"]:
            raise TrainV2BuildError(
                "train_v2 dựng lại khác bản V2 đã khóa "
                f"(expected {expected_sha256['output']}, received {actual}). "
                "Dữ liệu train mới phải ghi sang file/run mới, không ghi đè V2."
            )
    return payload, train, new_rows


def write_train_v2(
    project_dir: Path = PROJECT_DIR,
    expected_sha256: Mapping[str, str] | None = EXPECTED_SHA256,
    check_only: bool = False,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Build, verify and (unless ``check_only``) publish train_v2.

    Returns ``(status, baseline_train, new_rows)`` with status ``unchanged``,
    ``written`` or ``missing`` (only in ``check_only`` mode).
    """
    payload, train, new_rows = build_train_v2_bytes(project_dir, expected_sha256)
    output_path = project_dir / RELATIVE_PATHS["output"]
    if output_path.is_file():
        if output_path.read_bytes() == payload:
            return "unchanged", train, new_rows
        raise TrainV2BuildError(
            f"{output_path} đã tồn tại và khác bản dựng lại; không ghi đè. "
            "Kiểm tra file hiện có hoặc dùng tên file mới."
        )
    if check_only:
        return "missing", train, new_rows
    _atomic_write_if_changed(output_path, payload)
    return "written", train, new_rows


def _print_distribution(train: pd.DataFrame, new_rows: pd.DataFrame) -> None:
    combined = pd.concat([train, new_rows], ignore_index=True)
    summary = pd.DataFrame({
        "Train v1": train["news_type"].value_counts(),
        "Thêm mới": new_rows["news_type"].value_counts(),
        "Train v2": combined["news_type"].value_counts(),
    }).fillna(0).astype(int)
    print(summary.to_string())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true",
        help="Chỉ kiểm tra file train_v2 hiện có khớp bản dựng lại; không ghi.",
    )
    args = parser.parse_args(argv)

    try:
        status, train, new_rows = write_train_v2(check_only=args.check)
    except TrainV2BuildError as error:
        print(f"LỖI: {error}", file=sys.stderr)
        return 1

    print(f"Train v1: {len(train)} | thêm TRAIN_V2: {len(new_rows)} | train v2: {len(train) + len(new_rows)}")
    _print_distribution(train, new_rows)
    messages = {
        "unchanged": "train_v2.csv đã tồn tại và khớp byte-for-byte bản khóa; không ghi lại.",
        "written": "Đã ghi train_v2.csv (khớp hash bản khóa).",
        "missing": "Bản dựng lại hợp lệ nhưng train_v2.csv chưa tồn tại (chế độ --check, không ghi).",
    }
    print(messages[status])
    return 0 if status != "missing" else 1


if __name__ == "__main__":
    raise SystemExit(main())
