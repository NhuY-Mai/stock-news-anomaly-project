"""Training and validation helpers for the PhoBERT-CNN news classifier."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import math
import random
import re
import shutil
import tempfile
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch; hardware can still affect reproducibility."""
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _saved_best_metadata(epoch: Any, macro_f1: Any) -> dict[str, int | float]:
    if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 1:
        raise ValueError("Saved best epoch must be a positive integer.")
    if (
        isinstance(macro_f1, bool) or not isinstance(macro_f1, Real)
        or not math.isfinite(macro_f1) or not 0 <= macro_f1 <= 1
    ):
        raise ValueError("Saved validation Macro-F1 must be finite and between zero and one.")
    return {"epoch": int(epoch), "validation_macro_f1": float(macro_f1)}


VALIDATION_REVISION_ID = "validation_review_20261025_v2"
VALIDATION_REVISION_SOURCE_RUN = "20261024_153932_380173"
_REVIEW_LABEL_CHANGES = {
    "2243033b0d5b0474496fb850c24265ca1f2786be70d1f83f140db07746b4b16f": ("stock_market", "business_operations"),
    "1d85806da71bbed30f87fb08768b156742ad9bb28ec7debb0b5f343c2eea0098": ("stock_market", "business_operations"),
    "bcd95f1e82aea14779999077cfa9d1b50d67a61339ad9de7cab87876d69c5a97": ("financial_results", "business_operations"),
    "24c79e1c2ffee2f35c6e081a2e24c9c9191cf492bbd051c8364f58ff364a9464": ("financial_results", "business_operations"),
    "0924ce353ae49d611b69c2d14935624b6023ff60543e4208a02bdb20d6d65d62": ("stock_market", "governance_capital"),
    "43b8ec1ee1eb43780236646e6a91148b484882fc0a51493f3892249d75c2f74d": ("stock_market", "governance_capital"),
}
_REVIEW_CONTENT_CHANGES = {
    "0924ce353ae49d611b69c2d14935624b6023ff60543e4208a02bdb20d6d65d62": "prefix",
    "8a622ba716429d810d76c613fdb3b8d163c0018af47193379b2326c4287854d9": "suffix",
    "62029430c4b97773069ee32d35b42c142f819d89aed26c9dfcc55ebd17da95ab": "suffix",
}
_REVIEW_PENDING_IDS = {
    "f52e0fbda1db9bac08cce9aef325ec5418d0ebc1047ce94729357c513728e68f",
    "e98f7563f84e8a626295e73632ebd7679ae7f18961162b5fcabcf24ce90a8213",
}


def _review_row_sha256(row: dict[str, str]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_review_decisions(revision: dict[str, Any]) -> set[str]:
    """Account for both reviewed articles without authorizing a label change."""
    pending = revision.get("pending_reviews")
    resolved = revision.get("resolved_reviews", [])
    for name, decisions in (("pending", pending), ("resolved", resolved)):
        if (not isinstance(decisions, list)
                or any(not isinstance(item, dict) or not isinstance(item.get("news_id"), str)
                       or not isinstance(item.get("reason"), str) or not item["reason"].strip()
                       for item in decisions)):
            raise ValueError(f"The {name} review decisions require article IDs and human reasons.")
    article_ids = [item["news_id"] for item in pending + resolved]
    if len(article_ids) != 2 or set(article_ids) != _REVIEW_PENDING_IDS:
        raise ValueError("Both reviewed validation articles must be accounted for exactly once.")
    if any(item.get("news_type") != "financial_results"
           or item.get("decision") != "confirmed_unchanged"
           or item.get("reviewed_date") != "2026-10-25" for item in resolved):
        raise ValueError("Resolved reviews must record the approved unchanged financial-results labels.")
    return {item["news_id"] for item in resolved}


def validate_validation_revision(
    saved_config: dict[str, Any],
    *,
    input_sha256: dict[str, str],
    label_mapping: dict[str, Any],
    revision: dict[str, Any],
    validation_path: Path | str,
    source_run: str,
    run_config_sha256: str,
) -> str:
    """Replay only the approved validation edits back to the checkpoint's input.

    The original full-CSV hash authenticates unchanged rows, their order and all
    immutable fields. This is not a general option to ignore input mismatches.
    """
    from .cleaning_utils import REVIEWED_CONTENT_CORRECTIONS
    from .model_data_utils import MODEL_SPLIT_COLUMNS, build_phobert_input_text

    if not isinstance(revision, dict) or revision.get("revision_id") != VALIDATION_REVISION_ID:
        raise ValueError("Unknown or missing approved validation revision.")
    if source_run != VALIDATION_REVISION_SOURCE_RUN or revision.get("source_run") != source_run:
        raise ValueError("The validation revision belongs to a different saved run.")
    if revision.get("baseline_run_config_sha256") != run_config_sha256:
        raise ValueError("The validation revision does not match the original run configuration.")
    baseline_hashes = saved_config.get("input_sha256")
    if not isinstance(baseline_hashes, dict) or set(baseline_hashes) != set(input_sha256):
        raise ValueError("Saved input fingerprints are incomplete.")
    if (revision.get("baseline_input_sha256") != baseline_hashes
            or revision.get("revised_input_sha256") != input_sha256):
        raise ValueError("The validation revision input hashes do not match.")
    for name in ("train", "test"):
        path = f"data/splits/news_type/{name}.csv"
        if input_sha256.get(path) != baseline_hashes.get(path):
            raise ValueError("A validation-only revision cannot change train or test.")
    baseline_mapping = saved_config.get("label_mapping")
    if not isinstance(baseline_mapping, dict):
        raise ValueError("Saved label mapping is missing.")
    if (revision.get("baseline_label_policy_version") != "gold_price_industry_v1"
            or revision.get("revised_label_policy_version") != "company_awards_listing_v2"
            or baseline_mapping.get("label_policy_version") != "gold_price_industry_v1"):
        raise ValueError("The validation revision label policy is not approved.")
    revised_mapping = {**baseline_mapping, "label_policy_version": "company_awards_listing_v2"}
    if label_mapping != revised_mapping:
        raise ValueError("A validation revision cannot change label IDs or mapping semantics.")
    csv_format = revision.get("csv_format")
    if (not isinstance(csv_format, dict) or set(csv_format) != {"encoding", "lineterminator"}
            or csv_format["encoding"] not in {"utf-8", "utf-8-sig"}
            or csv_format["lineterminator"] not in {"\n", "\r\n"}):
        raise ValueError("The original validation CSV format is required.")
    resolved_ids = _validate_review_decisions(revision)
    payload = Path(validation_path).read_bytes()
    validation_key = "data/splits/news_type/validation.csv"
    if hashlib.sha256(payload).hexdigest() != input_sha256.get(validation_key):
        raise ValueError("Current validation bytes do not match their input fingerprint.")
    current = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
    if (current.columns.tolist() != MODEL_SPLIT_COLUMNS + ["split"] or current.empty
            or current["news_id"].eq("").any() or not current["news_id"].is_unique
            or not current["split"].eq("validation").all()):
        raise ValueError("The revised validation schema, article IDs or split tags are invalid.")
    changes = revision.get("changes")
    expected_ids = set(_REVIEW_LABEL_CHANGES) | set(_REVIEW_CONTENT_CHANGES)
    if (not isinstance(changes, list) or len(changes) != len(expected_ids)
            or any(not isinstance(item, dict) or not isinstance(item.get("news_id"), str) for item in changes)
            or {item.get("news_id") for item in changes} != expected_ids):
        raise ValueError("Only the eight approved validation articles may change.")
    before = current.copy(deep=True)
    positions = {news_id: index for index, news_id in enumerate(current["news_id"])}
    label2id = baseline_mapping["label2id"]
    for change in changes:
        news_id = change["news_id"]
        if news_id not in positions:
            raise ValueError("An approved validation article is missing.")
        position = positions[news_id]
        row = current.iloc[position].to_dict()
        fields = change.get("fields")
        expected_fields = set()
        if news_id in _REVIEW_LABEL_CHANGES:
            expected_fields.update(("news_type", "label_id"))
        if news_id in _REVIEW_CONTENT_CHANGES:
            expected_fields.update(("content", "input_text"))
        if not isinstance(fields, dict) or set(fields) != expected_fields:
            raise ValueError("The revision changes fields outside its approved scope.")
        if _review_row_sha256(row) != change.get("after_row_sha256"):
            raise ValueError("A revised validation row does not match its audit fingerprint.")
        old_row = dict(row)
        for column, values in fields.items():
            if (not isinstance(values, dict) or set(values) != {"before", "after"}
                    or not all(isinstance(value, str) for value in values.values())
                    or values["before"] == values["after"] or row[column] != values["after"]):
                raise ValueError("Revision field values must record the exact original and revised text.")
            old_row[column] = values["before"]
            before.at[position, column] = values["before"]
        if _review_row_sha256(old_row) != change.get("before_row_sha256"):
            raise ValueError("An original validation row does not match its audit fingerprint.")
        if news_id in _REVIEW_LABEL_CHANGES:
            old_label, new_label = _REVIEW_LABEL_CHANGES[news_id]
            if (old_row["news_type"] != old_label or row["news_type"] != new_label
                    or old_row["label_id"] != str(label2id[old_label])
                    or row["label_id"] != str(label2id[new_label])):
                raise ValueError("The revised label is not the user-approved correction.")
        if news_id in _REVIEW_CONTENT_CHANGES:
            old_content, new_content = old_row["content"], row["content"]
            approved_content = REVIEWED_CONTENT_CORRECTIONS[news_id]
            if (hashlib.sha256(old_content.encode("utf-8")).hexdigest() != approved_content["old_sha256"]
                    or hashlib.sha256(new_content.encode("utf-8")).hexdigest() != approved_content["new_sha256"]):
                raise ValueError("Article cleanup does not match the exact approved content fingerprints.")
            retained = (old_content.endswith(new_content) if _REVIEW_CONTENT_CHANGES[news_id] == "prefix"
                        else old_content.startswith(new_content))
            if not retained or not new_content or len(new_content) < len(old_content) / 2:
                raise ValueError("Approved cleanup may only trim the reviewed prefix or suffix.")
            for record in (row, old_row):
                expected_text = build_phobert_input_text(record["title"], record["summary"], record["content"])
                if record["input_text"] != expected_text:
                    raise ValueError("Revised model text must be rebuilt from the approved article content.")
    if not _REVIEW_PENDING_IDS.issubset(positions):
        raise ValueError("A reviewed validation article was removed.")
    for news_id in resolved_ids:
        for frame in (current, before):
            row = frame.iloc[positions[news_id]]
            if (row["news_type"] != "financial_results"
                    or row["label_id"] != str(label2id["financial_results"])):
                raise ValueError("A confirmed review must retain its original financial-results label.")
    original_payload = before.to_csv(index=False, lineterminator=csv_format["lineterminator"]).encode(csv_format["encoding"])
    if hashlib.sha256(original_payload).hexdigest() != baseline_hashes.get(validation_key):
        raise ValueError("Validation replay changed membership, order or unapproved article data.")
    return VALIDATION_REVISION_ID


def load_saved_run_metadata(
    model_dir: Path | str,
    report_dir: Path | str,
    *,
    input_sha256: dict[str, str],
    label_mapping: dict[str, Any],
    expected_config: dict[str, Any],
    validation_revision: dict[str, Any] | None = None,
    validation_path: Path | str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Read an explicitly selected compatible run without changing its artifacts."""
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    required_hash_paths = {
        "models/news_type_phobert_cnn/label_mapping.json",
        *(f"data/splits/news_type/{name}.csv" for name in ("train", "validation", "test")),
    }
    if (
        not isinstance(input_sha256, dict) or set(input_sha256) != required_hash_paths
        or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in input_sha256.values())
    ):
        raise ValueError("Current input_sha256 must contain the four complete input hashes.")
    if not isinstance(label_mapping, dict) or not label_mapping:
        raise ValueError("Provide the full current label mapping, including its policy version.")
    if not isinstance(expected_config, dict) or not expected_config:
        raise ValueError("Provide the preprocessing and architecture configuration to verify.")
    required_files = [
        model_dir / "best_model.pt", model_dir / "run_config.json",
        report_dir / "training_history.csv",
    ]
    if any(not path.is_file() for path in required_files):
        raise FileNotFoundError("The selected run is missing its checkpoint, configuration or history.")
    if any(not (model_dir / name).is_dir() for name in ("tokenizer", "backbone")):
        raise FileNotFoundError("The selected run is missing its saved tokenizer or backbone configuration.")
    config_path = model_dir / "run_config.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(saved_config, dict):
        raise ValueError("Saved run_config.json must contain a configuration object.")
    evaluation_revision = None
    if validation_revision is not None:
        if validation_path is None or report_dir.name != model_dir.name:
            raise ValueError("Approved revision review requires the original run and current validation file.")
        evaluation_revision = validate_validation_revision(
            saved_config, input_sha256=input_sha256, label_mapping=label_mapping,
            revision=validation_revision, validation_path=validation_path,
            source_run=model_dir.name, run_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        )
    else:
        if saved_config.get("input_sha256") != input_sha256:
            raise ValueError("Saved run input hashes do not match the current data and label mapping.")
        if saved_config.get("label_mapping") != label_mapping:
            raise ValueError("Saved run label mapping or label policy does not match.")
    for name, expected_value in expected_config.items():
        if name not in saved_config or saved_config[name] != expected_value:
            raise ValueError(f"Saved run configuration does not match {name}.")
    best = _saved_best_metadata(
        saved_config.get("best_epoch"), saved_config.get("best_validation_macro_f1"),
    )
    history = pd.read_csv(report_dir / "training_history.csv")
    history_columns = [
        "epoch", "train_loss", "train_accuracy", "train_macro_f1",
        "validation_loss", "validation_accuracy", "validation_macro_f1",
    ]
    if history.empty or not set(history_columns).issubset(history.columns):
        raise ValueError("Saved training history is empty or missing required metric columns.")
    numeric_history = history[history_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric_history.to_numpy(dtype=float)).all():
        raise ValueError("Saved training history contains non-finite metrics.")
    if numeric_history["epoch"].tolist() != list(range(1, len(history) + 1)):
        raise ValueError("Saved history epochs must be consecutive and start at one.")
    score_columns = [name for name in history_columns if name.endswith(("accuracy", "f1"))]
    if (
        (numeric_history[["train_loss", "validation_loss"]] < 0).any().any()
        or not (numeric_history[score_columns].ge(0) & numeric_history[score_columns].le(1)).all().all()
    ):
        raise ValueError("Saved history losses or classification scores are outside their valid range.")
    selected_row = numeric_history.loc[numeric_history["validation_macro_f1"].idxmax()]
    if (
        selected_row["epoch"] != best["epoch"]
        or not math.isclose(selected_row["validation_macro_f1"], best["validation_macro_f1"],
                            rel_tol=0, abs_tol=1e-10)
    ):
        raise ValueError("Saved best epoch or Macro-F1 does not match the training history.")
    history[history_columns] = numeric_history
    history.attrs.update(best_epoch=best["epoch"], best_validation_macro_f1=best["validation_macro_f1"])
    if evaluation_revision is not None:
        history.attrs["evaluation_revision"] = evaluation_revision
    return saved_config, history


def validate_saved_segmenter(saved_config: dict[str, Any], resource_hashes: dict[str, str]) -> None:
    """Refuse to encode saved-run inputs with different segmentation resources."""
    if not resource_hashes or saved_config.get("segmenter_sha256") != resource_hashes:
        raise ValueError("Saved run segmenter hashes do not match the current resources.")


def restore_saved_checkpoint(
    model: nn.Module,
    model_dir: Path | str,
    saved_config: dict[str, Any],
    device: torch.device | str,
) -> dict[str, int | float]:
    """Load verified saved weights for evaluation only; never resume an optimizer."""
    expected_best = _saved_best_metadata(
        saved_config.get("best_epoch"), saved_config.get("best_validation_macro_f1"),
    )
    checkpoint = torch.load(Path(model_dir) / "best_model.pt", map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Saved checkpoint is missing model_state_dict.")
    best = _saved_best_metadata(checkpoint.get("epoch"), checkpoint.get("validation_macro_f1"))
    if (
        best["epoch"] != expected_best["epoch"]
        or not math.isclose(best["validation_macro_f1"], expected_best["validation_macro_f1"],
                            rel_tol=0, abs_tol=1e-10)
    ):
        raise ValueError("Saved checkpoint epoch or Macro-F1 does not match its configuration.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return best


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scoped_test_file(directory: Path, relative_path: str) -> Path:
    path = (directory / relative_path).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("An evaluation artifact points outside its selected directory.")
    if not path.is_file():
        raise FileNotFoundError(f"Required saved evaluation artifact is missing: {relative_path}")
    return path


def _saved_model_asset_hashes(model_dir: Path) -> dict[str, str]:
    for name in ("tokenizer/tokenizer_config.json", "backbone/config.json"):
        _scoped_test_file(model_dir, name)
    assets = {}
    for folder in ("tokenizer", "backbone"):
        files = sorted(path for path in (model_dir / folder).rglob("*") if path.is_file())
        if not files:
            raise FileNotFoundError(f"Saved model assets are missing: {folder}")
        for path in files:
            relative = path.relative_to(model_dir).as_posix()
            assets[relative] = _artifact_sha256(_scoped_test_file(model_dir, relative))
    return assets


def _validate_test_manifest_context(
    model_dir: Path, input_sha256: dict[str, str], label_mapping: dict[str, Any],
    validation_revision_sha256: str | None, segmenter_sha256: dict[str, str],
    max_length: int, head_ratio: float,
) -> dict[str, Any]:
    required_paths = {
        "models/news_type_phobert_cnn/label_mapping.json",
        *(f"data/splits/news_type/{name}.csv" for name in ("train", "validation", "test")),
    }
    if (not isinstance(input_sha256, dict) or set(input_sha256) != required_paths
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in input_sha256.values())):
        raise ValueError("Test evaluation requires all four current input fingerprints.")
    label2id = label_mapping.get("label2id") if isinstance(label_mapping, dict) else None
    if (not isinstance(label2id, dict) or len(label2id) < 2
            or any(not isinstance(label, str) or not label for label in label2id)
            or any(type(value) is not int for value in label2id.values())
            or set(label2id.values()) != set(range(len(label2id)))):
        raise ValueError("Test evaluation requires a complete contiguous label mapping.")
    if validation_revision_sha256 is not None and (
        not isinstance(validation_revision_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", validation_revision_sha256)
    ):
        raise ValueError("The validation revision fingerprint is invalid.")
    if (isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 3
            or isinstance(head_ratio, bool) or not isinstance(head_ratio, Real)
            or not math.isfinite(head_ratio) or not 0 <= head_ratio <= 1):
        raise ValueError("Saved test preprocessing settings are invalid.")
    config_path = _scoped_test_file(model_dir, "run_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not isinstance(config.get("input_sha256"), dict):
        raise ValueError("The selected model configuration is incomplete.")
    for name in ("train", "test"):
        path = f"data/splits/news_type/{name}.csv"
        if config["input_sha256"].get(path) != input_sha256[path]:
            raise ValueError("The test evaluation model belongs to different train or test data.")
    old_mapping = config.get("label_mapping")
    if (not isinstance(old_mapping, dict)
            or {k: v for k, v in old_mapping.items() if k != "label_policy_version"}
            != {k: v for k, v in label_mapping.items() if k != "label_policy_version"}):
        raise ValueError("The test evaluation label mapping differs from the selected model.")
    if config.get("num_classes") != len(label2id):
        raise ValueError("The saved class count differs from the test label mapping.")
    if (config["input_sha256"] != input_sha256 or old_mapping != label_mapping) and validation_revision_sha256 is None:
        raise ValueError("Changed evaluation inputs require their approved revision fingerprint.")
    if config.get("max_length") != max_length or config.get("head_ratio") != head_ratio:
        raise ValueError("Test preprocessing differs from the saved model.")
    if (not isinstance(segmenter_sha256, dict) or not segmenter_sha256
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in segmenter_sha256.values())):
        raise ValueError("Test evaluation requires complete segmentation fingerprints.")
    validate_saved_segmenter(config, segmenter_sha256)
    _saved_best_metadata(config.get("best_epoch"), config.get("best_validation_macro_f1"))
    return config


def _verified_test_report_artifacts(
    report_dir: Path, test_data: pd.DataFrame, label2id: dict[str, int],
) -> tuple[dict[str, float], dict[str, str]]:
    required_columns = {"news_id", "source", "news_type", "label_id", "split"}
    if not isinstance(test_data, pd.DataFrame) or not required_columns.issubset(test_data.columns):
        raise ValueError("Test data is missing identity, source or ground-truth fields.")
    current = test_data.copy(deep=True)
    if current["label_id"].map(lambda value: isinstance(value, bool)).any():
        raise ValueError("Test label IDs must not be booleans.")
    current["label_id"] = pd.to_numeric(current["label_id"], errors="raise")
    _validate_review_frame(current, "test", label2id)
    payloads = {
        name: _scoped_test_file(report_dir, name).read_bytes()
        for name in ("test_metrics.json", "test_predictions.csv", "test_classification_report.csv")
    }
    predictions = pd.read_csv(io.BytesIO(payloads["test_predictions.csv"]), dtype=str, keep_default_na=False)
    expected_columns = [
        "news_id", "source", "news_type", "predicted_news_type", "predicted_probability", "is_correct",
    ]
    if predictions.columns.tolist() != expected_columns:
        raise ValueError("Test prediction columns are invalid.")
    for column in ("news_id", "source", "news_type"):
        if predictions[column].tolist() != current[column].astype(str).tolist():
            raise ValueError("Test predictions do not match the full current test order or labels.")
    predicted_ids = predictions["predicted_news_type"].map(label2id)
    if predicted_ids.isna().any():
        raise ValueError("Test predictions contain an unknown class.")
    scores = pd.to_numeric(predictions["predicted_probability"], errors="raise")
    if not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Test prediction scores must be finite probabilities.")
    correct = predictions["news_type"].eq(predictions["predicted_news_type"])
    if predictions["is_correct"].tolist() != correct.astype(str).tolist():
        raise ValueError("Test prediction correctness flags disagree with the labels.")
    metrics = json.loads(payloads["test_metrics.json"])
    expected_metric_names = {"loss", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"}
    if not isinstance(metrics, dict) or set(metrics) != expected_metric_names:
        raise ValueError("Test metrics are incomplete.")
    for name, value in metrics.items():
        if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                or value < 0 or (name != "loss" and value > 1)):
            raise ValueError("Test metrics contain invalid values.")
    truth, guessed = current["label_id"].to_numpy(dtype=int), predicted_ids.to_numpy(dtype=int)
    classes = list(range(len(label2id)))
    recomputed = {
        "accuracy": accuracy_score(truth, guessed),
        "macro_f1": f1_score(truth, guessed, labels=classes, average="macro", zero_division=0),
        "weighted_f1": f1_score(truth, guessed, labels=classes, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, guessed),
    }
    if any(not math.isclose(metrics[name], value, rel_tol=0, abs_tol=1e-10)
           for name, value in recomputed.items()):
        raise ValueError("Test metrics do not match the exported predictions.")
    labels = sorted(label2id, key=label2id.get)
    expected_report = pd.DataFrame(classification_report(
        truth, guessed, labels=classes, target_names=labels, output_dict=True, zero_division=0,
    )).T
    report = pd.read_csv(io.BytesIO(payloads["test_classification_report.csv"]), index_col=0)
    if (report.index.tolist() != expected_report.index.tolist()
            or report.columns.tolist() != expected_report.columns.tolist()
            or not np.allclose(report.to_numpy(dtype=float), expected_report.to_numpy(dtype=float),
                               rtol=0, atol=1e-10)):
        raise ValueError("The per-class test report does not match the exported predictions.")
    fingerprints = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    return metrics, fingerprints


def build_test_evaluation_manifest(
    run_model_dir: Path | str, report_dir: Path | str,
    *, input_sha256: dict[str, str], label_mapping: dict[str, Any],
    validation_revision_sha256: str | None, segmenter_sha256: dict[str, str],
    max_length: int, head_ratio: float, test_data: pd.DataFrame,
) -> dict[str, Any]:
    """Describe already exported real test results; never run a model or write files.

    The caller must first restore a compatible model and evaluate test. Loss is
    checked for finiteness; the four classification scores are recomputed from
    the full exported prediction table, not trusted because a file exists.
    """
    model_dir, report_dir = Path(run_model_dir).resolve(), Path(report_dir).resolve()
    config = _validate_test_manifest_context(
        model_dir, input_sha256, label_mapping, validation_revision_sha256,
        segmenter_sha256, max_length, head_ratio,
    )
    metrics, artifact_hashes = _verified_test_report_artifacts(report_dir, test_data, label_mapping["label2id"])
    manifest = {
        "schema_version": 1, "evaluated_split": "test", "test_evaluated": True,
        "source_run": model_dir.name, "checkpoint_epoch": int(config["best_epoch"]),
        "checkpoint_sha256": _artifact_sha256(_scoped_test_file(model_dir, "best_model.pt")),
        "run_config_sha256": _artifact_sha256(_scoped_test_file(model_dir, "run_config.json")),
        "model_assets_sha256": _saved_model_asset_hashes(model_dir),
        "input_sha256": input_sha256, "label_mapping": label_mapping,
        "validation_revision_sha256": validation_revision_sha256,
        "segmenter_sha256": segmenter_sha256, "max_length": int(max_length),
        "head_ratio": float(head_ratio), "test_count": int(len(test_data)),
        "metrics": metrics, "artifact_sha256": artifact_hashes,
    }
    return json.loads(json.dumps(manifest, ensure_ascii=False))


def verify_test_evaluation_manifest(
    run_model_dir: Path | str, report_dir: Path | str,
    *, input_sha256: dict[str, str], label_mapping: dict[str, Any],
    validation_revision_sha256: str | None, segmenter_sha256: dict[str, str],
    max_length: int, head_ratio: float, test_data: pd.DataFrame,
) -> dict[str, Any]:
    """Require notebook 05's fingerprint-bound test export before bulk inference."""
    try:
        manifest_path = _scoped_test_file(Path(report_dir), "test_evaluation_config.json")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Run notebook 05 section 14, including its test provenance export, before notebook 06. "
            "Missing test_evaluation_config.json."
        ) from None
    observed = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expected = build_test_evaluation_manifest(
        run_model_dir, report_dir, input_sha256=input_sha256, label_mapping=label_mapping,
        validation_revision_sha256=validation_revision_sha256, segmenter_sha256=segmenter_sha256,
        max_length=max_length, head_ratio=head_ratio, test_data=test_data,
    )
    canonical = lambda payload: json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if not isinstance(observed, dict) or canonical(observed) != canonical(expected):
        raise ValueError("Saved test provenance does not match the selected model, data or report artifacts.")
    return observed


def make_optimizer(
    model: nn.Module,
    backbone_lr: float = 2e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> AdamW:
    """Use separate learning rates and exclude biases/normalization from decay."""
    if not hasattr(model, "backbone"):
        raise ValueError("The model must expose its PhoBERT encoder as backbone.")
    if any(not math.isfinite(rate) or rate <= 0 for rate in (backbone_lr, head_lr)):
        raise ValueError("Learning rates must be finite and positive.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative.")

    normalization_types = (
        nn.LayerNorm,
        nn.GroupNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        getattr(nn, "RMSNorm", nn.LayerNorm),
    )
    normalization_parameters = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, normalization_types)
        for parameter in module.parameters(recurse=False)
    }
    backbone_parameters = {id(parameter) for parameter in model.backbone.parameters()}
    grouped_parameters: dict[tuple[bool, bool], list[nn.Parameter]] = {}

    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = id(parameter) in backbone_parameters
        is_bias = parameter_name == "bias" or parameter_name.endswith(".bias")
        use_decay = not is_bias and id(parameter) not in normalization_parameters
        grouped_parameters.setdefault((is_backbone, use_decay), []).append(parameter)

    if not grouped_parameters:
        raise ValueError("The model has no trainable parameters.")

    parameter_groups = [
        {
            "params": parameters,
            "lr": backbone_lr if is_backbone else head_lr,
            "weight_decay": weight_decay if use_decay else 0.0,
        }
        for (is_backbone, use_decay), parameters in grouped_parameters.items()
    ]
    return AdamW(parameter_groups)


def make_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
) -> LambdaLR:
    """Warm up linearly, then decay to zero; step after each optimizer update."""
    if not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer.")
    if not math.isfinite(warmup_ratio) or not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be between 0 (inclusive) and 1.")
    warmup_steps = int(total_steps * warmup_ratio)

    def learning_rate_factor(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        decay_steps = total_steps - warmup_steps
        return max(0.0, (total_steps - current_step) / max(1, decay_steps))

    return LambdaLR(optimizer, lr_lambda=learning_rate_factor)


def _prepare_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device | str,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    required_fields = ("input_ids", "attention_mask", "labels")
    if any(field not in batch for field in required_fields):
        raise ValueError("Each batch must contain input_ids, attention_mask and labels.")
    if any(not isinstance(batch[field], torch.Tensor) for field in required_fields):
        raise TypeError("Batch inputs and labels must be PyTorch tensors.")

    labels = batch["labels"]
    integer_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    if labels.ndim != 1 or labels.dtype not in integer_types:
        raise ValueError("labels must be a one-dimensional integer tensor.")
    if labels.numel() == 0:
        raise ValueError("A batch must contain at least one article.")
    if bool(((labels < 0) | (labels >= num_classes)).any()):
        raise ValueError("A label is outside the configured class range.")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have matching [batch, length] shapes.")
    if input_ids.shape[0] != labels.shape[0]:
        raise ValueError("The number of input rows must match the number of labels.")

    return {
        "input_ids": input_ids.to(device, non_blocking=True),
        "attention_mask": attention_mask.to(device, non_blocking=True),
        "labels": labels.to(device, dtype=torch.long, non_blocking=True),
    }


def _check_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("The model must return a logits tensor.")
    if logits.shape != (labels.shape[0], num_classes):
        raise ValueError("Model logits must have shape [batch, num_classes].")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Model logits contain non-finite values.")


def _check_loss(loss: torch.Tensor) -> None:
    if loss.ndim != 0:
        raise ValueError("The criterion must return a scalar loss (use reduction='mean').")
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("The loss is not finite; no optimizer update was made.")


def _summarize_predictions(
    logits_batches: list[torch.Tensor],
    label_batches: list[torch.Tensor],
    criterion: nn.Module,
    device: torch.device | str,
    num_classes: int,
) -> dict[str, Any]:
    if not label_batches:
        raise ValueError("The data loader is empty.")

    logits = torch.cat(logits_batches).float()
    labels = torch.cat(label_batches)
    predictions = logits.argmax(dim=1)
    # One complete calculation preserves the weighted-CE denominator across batches.
    with torch.no_grad():
        loss = criterion(logits.to(device), labels.to(device))
    _check_loss(loss)

    label_values = labels.numpy()
    prediction_values = predictions.numpy()
    class_ids = list(range(num_classes))
    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy_score(label_values, prediction_values)),
        "macro_f1": float(
            f1_score(label_values, prediction_values, labels=class_ids, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(label_values, prediction_values, labels=class_ids, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(label_values, prediction_values)),
        "labels": label_values.tolist(),
        "predictions": prediction_values.tolist(),
        "probabilities": torch.softmax(logits, dim=1).numpy(),
    }


def evaluate_classifier(
    model: nn.Module,
    loader: Any,
    device: torch.device | str,
    num_classes: int,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate a split without gradients; return probabilities for reporting only."""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")
    loss_function = criterion if criterion is not None else nn.CrossEntropyLoss()
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            prepared_batch = _prepare_batch(batch, device, num_classes)
            logits = model(
                input_ids=prepared_batch["input_ids"],
                attention_mask=prepared_batch["attention_mask"],
            )
            labels = prepared_batch["labels"]
            _check_logits(logits, labels, num_classes)
            logits_batches.append(logits.detach().cpu())
            label_batches.append(labels.detach().cpu())

    return _summarize_predictions(logits_batches, label_batches, loss_function, device, num_classes)


def _validate_review_frame(
    frame: pd.DataFrame, split: str, label2id: dict[str, int],
) -> None:
    required = {"news_id", "news_type", "label_id", "split"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError(f"{split} data must contain {sorted(required)}.")
    if frame.empty or not frame["split"].eq(split).all():
        raise ValueError(f"Review requires non-empty {split} data, not another split.")
    article_ids = frame["news_id"]
    if article_ids.isna().any() or article_ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{split} news_id values must not be empty.")
    if article_ids.astype(str).duplicated().any():
        raise ValueError(f"{split} news_id values must be unique.")
    expected_labels = frame["news_type"].map(label2id)
    if expected_labels.isna().any() or not expected_labels.eq(frame["label_id"]).all():
        raise ValueError(f"{split} labels disagree with label2id.")


def _validate_review_predictions(
    validation_data: pd.DataFrame, results: dict[str, Any], num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check positional labels; the caller must also guarantee article order."""
    labels = np.asarray(results.get("labels", []))
    predictions = np.asarray(results.get("predictions", []))
    expected_shape = (len(validation_data),)
    for name, values in (("labels", labels), ("predictions", predictions)):
        if values.shape != expected_shape or values.dtype.kind not in "iu":
            raise ValueError(f"Review {name} must be an aligned integer vector.")
        if ((values < 0) | (values >= num_classes)).any():
            raise ValueError(f"Review {name} contain an unknown class ID.")
    if not np.array_equal(labels, validation_data["label_id"].to_numpy()):
        raise ValueError("Validation result labels do not match the data order.")
    probabilities = np.asarray(results.get("probabilities", []), dtype=float)
    if probabilities.shape != (len(validation_data), num_classes):
        raise ValueError("Review probabilities must have shape [articles, classes].")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Review probabilities must be finite and between zero and one.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Review probability rows must sum to one.")
    if not np.array_equal(probabilities.argmax(axis=1), predictions):
        raise ValueError("Review predictions disagree with probability argmax.")
    return labels, predictions, probabilities


def build_validation_review(
    validation_data: pd.DataFrame,
    train_data: pd.DataFrame,
    validation_results: dict[str, Any],
    label2id: dict[str, int],
    encodings: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe validation errors without training, relabeling or reading test.

    Inputs must follow the same article order as a non-shuffled validation loader.
    Softmax values are uncalibrated scores, not verified label-correctness estimates.
    The returned tables are class summary, wrong articles and directed confusions.
    """
    if (
        not label2id or len(label2id) < 2
        or any(not isinstance(label, str) or not label for label in label2id)
        or any(type(value) is not int for value in label2id.values())
        or set(label2id.values()) != set(range(len(label2id)))
    ):
        raise ValueError("label2id must map at least two names to contiguous integer IDs.")
    _validate_review_frame(train_data, "train", label2id)
    _validate_review_frame(validation_data, "validation", label2id)
    if set(train_data["news_id"].astype(str)) & set(validation_data["news_id"].astype(str)):
        raise ValueError("Train and validation article IDs must be disjoint.")
    labels, predictions, probabilities = _validate_review_predictions(
        validation_data, validation_results, len(label2id),
    )
    ordered_labels = [label for label, _ in sorted(label2id.items(), key=lambda item: item[1])]
    articles = validation_data.copy().reset_index(drop=True)
    articles["predicted_news_type"] = [ordered_labels[index] for index in predictions]
    articles["predicted_probability"] = probabilities.max(axis=1)
    articles["assigned_label_probability"] = probabilities[np.arange(len(labels)), labels]
    top_two = np.sort(probabilities, axis=1)[:, -2:]
    articles["probability_margin"] = top_two[:, 1] - top_two[:, 0]
    if encodings is not None:
        if len(encodings) != len(articles):
            raise ValueError("Token statistics must match the validation article order.")
        lengths = pd.DataFrame(encodings)[["original_token_count", "retained_token_count"]]
        if (
            lengths.isna().any().any()
            or any(not pd.api.types.is_integer_dtype(lengths[column]) for column in lengths)
            or (lengths <= 0).any().any()
            or (lengths["retained_token_count"] > lengths["original_token_count"]).any()
        ):
            raise ValueError("Token counts must be positive integers with retained <= original.")
        articles[lengths.columns] = lengths.to_numpy()
        articles["was_truncated"] = lengths["original_token_count"].gt(lengths["retained_token_count"])
    errors = articles.loc[labels != predictions].copy()
    pairs = (
        errors.groupby(["news_type", "predicted_news_type"], sort=True).size()
        .rename("article_count").reset_index()
        .sort_values(["article_count", "news_type", "predicted_news_type"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(label2id))), zero_division=0,
    )
    summary = pd.DataFrame({
        "news_type": ordered_labels,
        "train_count": train_data["news_type"].value_counts().reindex(ordered_labels, fill_value=0).to_numpy(),
        "validation_count": support,
        "false_negatives": np.bincount(labels[labels != predictions], minlength=len(label2id)),
        "false_positives": np.bincount(predictions[labels != predictions], minlength=len(label2id)),
        "precision": precision, "recall": recall, "f1": class_f1,
    })
    dominant = pairs.drop_duplicates("news_type").rename(columns={
        "predicted_news_type": "most_confused_with", "article_count": "confusion_count",
    })
    summary = summary.merge(dominant, on="news_type", how="left", validate="one_to_one")
    summary["most_confused_with"] = summary["most_confused_with"].fillna("")
    summary["confusion_count"] = summary["confusion_count"].fillna(0).astype(int)
    summary = summary.sort_values(["f1", "train_count", "news_type"], kind="stable").reset_index(drop=True)
    errors = errors.sort_values("predicted_probability", ascending=False, kind="stable").reset_index(drop=True)
    return summary, errors, pairs


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: Optimizer,
    scheduler: LambdaLR | None,
    criterion: nn.Module,
    device: torch.device | str,
    num_classes: int,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Run FP32 updates, gradient clipping and one scheduler step per batch."""
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive.")

    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    model.train()
    for batch in loader:
        prepared_batch = _prepare_batch(batch, device, num_classes)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            input_ids=prepared_batch["input_ids"],
            attention_mask=prepared_batch["attention_mask"],
        )
        labels = prepared_batch["labels"]
        _check_logits(logits, labels, num_classes)
        loss = criterion(logits, labels)
        _check_loss(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        logits_batches.append(logits.detach().cpu())
        label_batches.append(labels.detach().cpu())

    metrics = _summarize_predictions(logits_batches, label_batches, criterion, device, num_classes)
    return {name: metrics[name] for name in ("loss", "accuracy", "macro_f1")}


def fit_classifier(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    optimizer: Optimizer,
    scheduler: LambdaLR | None,
    criterion: nn.Module,
    device: torch.device | str,
    num_classes: int,
    checkpoint_path: Path | str,
    max_epochs: int = 8,
    patience: int = 2,
    min_delta: float = 1e-4,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Select by validation Macro-F1, stop on a plateau and restore the best weights."""
    if not isinstance(max_epochs, int) or max_epochs <= 0:
        raise ValueError("max_epochs must be a positive integer.")
    if not isinstance(patience, int) or patience <= 0:
        raise ValueError("patience must be a positive integer.")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and non-negative.")

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_rows: list[dict[str, Any]] = []
    best_macro_f1 = -math.inf
    significant_macro_f1 = -math.inf
    epochs_without_improvement = 0
    model.to(device)

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, num_classes
        )
        validation_metrics = evaluate_classifier(
            model, validation_loader, device, num_classes, criterion=criterion
        )
        validation_macro_f1 = validation_metrics["macro_f1"]
        is_best = validation_macro_f1 > best_macro_f1
        if is_best:
            best_macro_f1 = validation_macro_f1
            checkpoint = {
                "model_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "epoch": epoch,
                "validation_macro_f1": float(validation_macro_f1),
            }
            torch.save(checkpoint, checkpoint_path)

        if validation_macro_f1 > significant_macro_f1 + min_delta:
            significant_macro_f1 = validation_macro_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": validation_metrics["loss"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_macro_f1,
                "is_best": is_best,
            }
        )
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"validation loss={validation_metrics['loss']:.4f} | "
            f"validation Macro-F1={validation_macro_f1:.4f}"
            + (" | best checkpoint saved" if is_best else "")
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs; restoring the best checkpoint.")
            break

    best_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    history = pd.DataFrame(history_rows)
    history.attrs["best_epoch"] = best_checkpoint["epoch"]
    history.attrs["best_validation_macro_f1"] = best_checkpoint["validation_macro_f1"]
    return history


_TRIAL_PARAMETERS = ("backbone_lr", "head_lr", "dropout", "weight_decay")
_VALIDATION_METRICS = ("loss", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")


def _validate_trial_configs(trial_configs: Any) -> list[dict[str, Any]]:
    """Reject ambiguous or unsafe trial definitions before creating any files."""
    if not isinstance(trial_configs, (list, tuple)) or not 1 <= len(trial_configs) <= 6:
        raise ValueError("Provide between one and six fixed trial configurations.")
    configs: list[dict[str, Any]] = []
    trial_ids: set[str] = set()
    reserved_names = {"CON", "PRN", "AUX", "NUL"}
    reserved_names.update(f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10))
    for config in trial_configs:
        if not isinstance(config, dict) or set(config) != {"trial_id", *_TRIAL_PARAMETERS}:
            raise ValueError("Each trial requires trial_id and exactly the four study parameters.")
        trial_id = config["trial_id"]
        if (
            not isinstance(trial_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", trial_id) is None
            or trial_id.upper() in reserved_names
        ):
            raise ValueError("trial_id must be a safe, portable directory name of at most 64 characters.")
        if trial_id.casefold() in trial_ids:
            raise ValueError(f"Duplicate trial_id: {trial_id}")
        trial_ids.add(trial_id.casefold())
        normalized = {"trial_id": trial_id}
        for parameter in _TRIAL_PARAMETERS:
            value = config[parameter]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{parameter} must be a finite number.")
            normalized[parameter] = float(value)
        if min(normalized["backbone_lr"], normalized["head_lr"]) <= 0:
            raise ValueError("Learning rates must be positive.")
        if not 0 <= normalized["dropout"] < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1.")
        if normalized["weight_decay"] < 0:
            raise ValueError("weight_decay must be non-negative.")
        configs.append(normalized)
    return configs


def rank_parameter_trials(results: pd.DataFrame) -> pd.DataFrame:
    """Rank validation results; never use the test split to select a trial."""
    required_columns = ("trial_id", "validation_macro_f1", "validation_accuracy", "validation_loss")
    if not isinstance(results, pd.DataFrame) or any(column not in results for column in required_columns):
        raise ValueError("Trial results must contain IDs and the three validation ranking metrics.")
    if results["trial_id"].isna().any() or results["trial_id"].duplicated().any():
        raise ValueError("Trial result IDs must be present and unique.")
    if not results["trial_id"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("Trial result IDs must be strings.")
    for metric in required_columns[1:]:
        values = pd.to_numeric(results[metric], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any():
            raise ValueError(f"{metric} must contain finite, non-negative numbers.")
        if metric != "validation_loss" and (values > 1).any():
            raise ValueError(f"{metric} must be between 0 and 1.")
    ranked = results.copy()
    for metric in required_columns[1:]:
        ranked[metric] = pd.to_numeric(ranked[metric])
    ranked = ranked.sort_values(
        list(required_columns[1:]) + ["trial_id"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["is_selected"] = ranked["rank"].eq(1)
    return ranked


def _write_study_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(path.name + ".part")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_study_results(path: Path, results: pd.DataFrame) -> None:
    temporary_path = path.with_name(path.name + ".part")
    try:
        results.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _retain_study_checkpoint(source: Path, destination: Path) -> None:
    """Copy locally staged weights before atomically replacing the study winner."""
    temporary_path = destination.with_name(destination.name + ".part")
    try:
        shutil.copyfile(source, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fit_parameter_trial(
    config: dict[str, Any],
    model_factory: Callable,
    loader_factory: Callable,
    criterion_factory: Callable,
    device: torch.device | str,
    settings: dict[str, Any],
    checkpoint_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Own all trial objects so no model or optimizer survives between trials."""
    model = optimizer = scheduler = criterion = train_loader = validation_loader = metrics = None
    try:
        seed_everything(settings["seed"])
        model = model_factory(dict(config))
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return a PyTorch module.")
        model.to(device)
        train_loader, validation_loader = loader_factory(dict(config))
        if len(train_loader) == 0 or len(validation_loader) == 0:
            raise ValueError("Both training and validation loaders must be non-empty.")
        criterion = criterion_factory(dict(config))
        if not isinstance(criterion, nn.Module):
            raise TypeError("criterion_factory must return a PyTorch loss module.")
        criterion.to(device)
        optimizer = make_optimizer(
            model, config["backbone_lr"], config["head_lr"], config["weight_decay"]
        )
        scheduler = make_scheduler(
            optimizer, len(train_loader) * settings["max_epochs"], settings["warmup_ratio"]
        )
        history = fit_classifier(
            model, train_loader, validation_loader, optimizer, scheduler, criterion,
            device, settings["num_classes"], checkpoint_path,
            max_epochs=settings["max_epochs"], patience=settings["patience"],
            min_delta=settings["min_delta"],
        )
        metrics = evaluate_classifier(
            model, validation_loader, device, settings["num_classes"], criterion=criterion
        )
        result = dict(config)
        for name in _VALIDATION_METRICS:
            value = float(metrics[name])
            if not math.isfinite(value) or value < 0 or (name != "loss" and value > 1):
                raise ValueError(f"The restored checkpoint has an invalid validation {name}.")
            result[f"validation_{name}"] = value
        if not math.isclose(
            result["validation_macro_f1"], history.attrs["best_validation_macro_f1"], abs_tol=1e-8
        ):
            raise ValueError("Restored-checkpoint Macro-F1 differs from the selected epoch.")
        result.update(best_epoch=int(history.attrs["best_epoch"]), epochs_run=len(history))
        return result, history
    finally:
        model = optimizer = scheduler = criterion = train_loader = validation_loader = metrics = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _validate_study_settings(
    seed: int, num_classes: int, max_epochs: int, patience: int,
    min_delta: float, warmup_ratio: float,
) -> dict[str, Any]:
    for name, value in (("seed", seed), ("num_classes", num_classes), ("max_epochs", max_epochs), ("patience", patience)):
        minimum = 0 if name == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    if seed >= 2**32:
        raise ValueError("seed must be less than 2**32.")
    for name, value in (("min_delta", min_delta), ("warmup_ratio", warmup_ratio)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if warmup_ratio >= 1:
        raise ValueError("warmup_ratio must be less than 1.")
    return {
        "seed": seed, "num_classes": num_classes, "max_epochs": max_epochs,
        "patience": patience, "min_delta": float(min_delta), "warmup_ratio": float(warmup_ratio),
    }


def run_parameter_study(
    trial_configs: list[dict[str, Any]],
    model_factory: Callable,
    loader_factory: Callable,
    criterion_factory: Callable,
    device: torch.device | str,
    num_classes: int,
    model_dir: Path | str,
    report_dir: Path | str,
    seed: int = 42,
    max_epochs: int = 8,
    patience: int = 2,
    min_delta: float = 1e-4,
    warmup_ratio: float = 0.1,
    context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compare up to six fresh runs on fixed train/validation data and keep one winner.

    Validation Macro-F1 is the primary selection criterion; accuracy, loss and the
    trial ID break ties. Completed trials and the current winner remain on disk if
    a later trial fails. These artifacts are not an optimizer-resume checkpoint.
    """
    configs = _validate_trial_configs(trial_configs)
    settings = _validate_study_settings(seed, num_classes, max_epochs, patience, min_delta, warmup_ratio)
    if not all(callable(factory) for factory in (model_factory, loader_factory, criterion_factory)):
        raise TypeError("Each trial factory must be callable.")
    if context is not None and not isinstance(context, dict):
        raise TypeError("context must be a JSON-serializable dictionary.")
    study_context = {**settings, "trials": configs, "context": context or {}}
    # Validate metadata now, not after an expensive run or a partial directory write.
    study_context = json.loads(json.dumps(study_context, allow_nan=False))
    model_dir, report_dir = Path(model_dir).resolve(), Path(report_dir).resolve()
    if model_dir == report_dir or model_dir in report_dir.parents or report_dir in model_dir.parents:
        raise ValueError("Model and report directories must be separate, non-nested paths.")
    for directory in (model_dir, report_dir):
        if directory.exists():
            raise FileExistsError(f"Study output directory already exists: {directory}")
    model_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    _write_study_json(report_dir / "study_context.json", study_context)
    completed_rows: list[dict[str, Any]] = []
    selected_trial_id: str | None = None
    checkpoint_path = model_dir / "best_model.pt"
    for trial_number, config in enumerate(configs, start=1):
        trial_id = config["trial_id"]
        print(f"Trial {trial_number}/{len(configs)}: {trial_id}")
        trial_report_dir = report_dir / "trials" / trial_id
        trial_report_dir.mkdir(parents=True, exist_ok=False)
        _write_study_json(trial_report_dir / "trial_config.json", config)
        with tempfile.TemporaryDirectory(prefix="phobert-study-") as temporary_directory:
            trial_checkpoint = Path(temporary_directory) / "best_model.pt"
            result, history = _fit_parameter_trial(
                config, model_factory, loader_factory, criterion_factory, device, settings, trial_checkpoint
            )
            history.to_csv(trial_report_dir / "training_history.csv", index=False)
            validation_metrics = {
                name: result[f"validation_{name}"] for name in _VALIDATION_METRICS
            }
            _write_study_json(trial_report_dir / "validation_metrics.json", validation_metrics)
            ranked = rank_parameter_trials(pd.DataFrame([*completed_rows, result]))
            winner_id = str(ranked.iloc[0]["trial_id"])
            if winner_id != selected_trial_id:
                _retain_study_checkpoint(trial_checkpoint, checkpoint_path)
                winner_metadata = {
                    "trial_id": trial_id, "config": config, "metrics": validation_metrics,
                    "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
                    "checkpoint_filename": checkpoint_path.name, "study_context": study_context,
                }
                _write_study_json(model_dir / "best_trial.json", winner_metadata)
                selected_trial_id = winner_id
            completed_rows.append(result)
            _write_study_results(report_dir / "study_results.csv", ranked)
    ranked.attrs.update(
        best_trial_id=selected_trial_id,
        best_config=next(config for config in configs if config["trial_id"] == selected_trial_id),
        checkpoint_path=str(checkpoint_path), model_dir=str(model_dir), report_dir=str(report_dir),
    )
    return ranked
