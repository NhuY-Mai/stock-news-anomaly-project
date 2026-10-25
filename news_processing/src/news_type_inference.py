"""Inference-only helpers that keep human news-type decisions authoritative."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .annotation_utils import (
    ANNOTATION_COLUMNS,
    NEWS_COLUMNS,
    refresh_annotation_content,
)
from .model_data_utils import build_phobert_input_text


PREDICTION_COLUMNS = ["news_id", "news_type", "prediction_score"]
OUTPUT_COLUMNS = ["news_id", "news_type", "label_source", "prediction_score"]
_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _ordered_labels(label2id: dict[str, int]) -> list[str]:
    if (
        not isinstance(label2id, dict) or len(label2id) < 2
        or any(not isinstance(name, str) or not name.strip() for name in label2id)
        or any(type(value) is not int for value in label2id.values())
        or set(label2id.values()) != set(range(len(label2id)))
    ):
        raise ValueError("label2id must map nonempty labels to contiguous integer IDs.")
    return sorted(label2id, key=label2id.get)


def _validate_ids(frame: pd.DataFrame, name: str) -> None:
    if "news_id" not in frame or any(
        not isinstance(value, str) or not value.strip() for value in frame["news_id"]
    ) or not frame["news_id"].is_unique:
        raise ValueError(f"{name} must contain unique, nonempty string news_id values.")


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or (
        not isinstance(value, (list, tuple, dict, np.ndarray)) and bool(pd.isna(value))
    )


def build_inference_plan(
    clean_news: pd.DataFrame,
    annotations: pd.DataFrame,
    excluded_records: pd.DataFrame,
    duplicate_merge_log: pd.DataFrame,
    label2id: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition clean articles into confirmed manual, unannotated and review hold.

    Annotation aliases follow the same cleaning lineage used in notebook 04.
    Unresolved/excluded human decisions and conflicting duplicate labels never
    become model targets. All input frames remain unchanged.
    """
    _ordered_labels(label2id)
    if clean_news.empty or clean_news.columns.tolist() != NEWS_COLUMNS:
        raise ValueError("Clean news must contain the eight expected fields in their saved order.")
    _validate_ids(clean_news, "Clean news")
    if clean_news["url"].duplicated().any() or any(
        not isinstance(url, str) or not url.strip() for url in clean_news["url"]
    ):
        raise ValueError("Clean news URLs must be unique before resolving annotation lineage.")
    required = set(NEWS_COLUMNS + ANNOTATION_COLUMNS)
    if not required.issubset(annotations.columns):
        raise ValueError("Annotations are missing required article or human-decision fields.")
    _validate_ids(annotations, "Annotations")
    if not annotations.index.is_unique:
        raise ValueError("Annotation row indices must be unique before resolving manual decisions.")
    if any(not isinstance(value, str) for value in annotations[ANNOTATION_COLUMNS].to_numpy().flat):
        raise ValueError("Read annotation decision fields as strings without missing-value conversion.")
    if not annotations["annotation_status"].isin({"", "labeled", "review_required", "excluded"}).all():
        raise ValueError("Annotations contain an unknown annotation_status.")
    if not {"kept_news_id", "merged_news_ids"}.issubset(duplicate_merge_log.columns):
        raise ValueError("The duplicate log is missing annotation-lineage columns.")
    if not {"news_id", "exclusion_reason"}.issubset(excluded_records.columns):
        raise ValueError("The excluded-article log is missing identity or reason columns.")
    lineage: dict[str, str] = {}
    clean_ids = set(clean_news["news_id"])
    for group in duplicate_merge_log.itertuples(index=False):
        for alias in str(group.merged_news_ids).split("|"):
            if not alias or not isinstance(group.kept_news_id, str) or not group.kept_news_id:
                raise ValueError("Duplicate lineage contains a blank article ID.")
            if alias in clean_ids and alias != group.kept_news_id:
                raise ValueError("Duplicate lineage cannot redirect an article that still exists in clean news.")
            if alias in lineage and lineage[alias] != group.kept_news_id:
                raise ValueError("An annotation alias maps to conflicting canonical articles.")
            lineage[alias] = group.kept_news_id
    if annotations.empty:
        refreshed = annotations.copy(deep=True)
        audit = pd.DataFrame(columns=[
            "news_id", "canonical_news_id", "training_news_type", "content_changed",
            "content_issue", "training_eligible",
        ])
    else:
        refreshed, audit = refresh_annotation_content(
            annotations, clean_news, excluded_records, duplicate_merge_log,
        )
    finalized = annotations["annotation_status"].eq("labeled")
    if not audit.loc[finalized, "training_news_type"].isin(label2id).all():
        raise ValueError("A finalized manual annotation has an empty or unknown news_type.")
    plan = pd.DataFrame({
        "news_id": clean_news["news_id"].tolist(),
        "news_type": "", "label_source": "unlabeled", "prediction_score": "",
    }, columns=OUTPUT_COLUMNS)
    positions = {news_id: index for index, news_id in enumerate(plan["news_id"])}
    linked = audit.loc[audit["canonical_news_id"].isin(positions)]
    for canonical_id, group in linked.groupby("canonical_news_id", sort=False):
        decisions = refreshed.loc[group.index, ANNOTATION_COLUMNS]
        unfinished = decisions["annotation_status"].isin({"review_required", "excluded"})
        partial = decisions["annotation_status"].eq("") & decisions[
            ["news_type", "annotator_note"]
        ].apply(lambda column: column.str.strip().ne("")).any(axis=1)
        confirmed = group.loc[decisions["annotation_status"].eq("labeled"), "training_news_type"]
        blocked = (
            unfinished.any() or partial.any() or confirmed.nunique() > 1
            or group["content_issue"].eq("DUPLICATE_LABEL_CONFLICT").any()
        )
        position = positions[canonical_id]
        if blocked:
            plan.at[position, "label_source"] = "review_required"
        elif not confirmed.empty:
            plan.at[position, "news_type"] = confirmed.iloc[0]
            plan.at[position, "label_source"] = "manual"
    targets = plan["label_source"].eq("unlabeled").to_numpy()
    articles = clean_news.loc[targets]
    inference_inputs = pd.DataFrame({
        "news_id": articles["news_id"].tolist(),
        "input_text": [
            build_phobert_input_text(title, summary, content)
            for title, summary, content in zip(articles["title"], articles["summary"], articles["content"])
        ],
    }, dtype=object)
    if any(not text.strip() for text in inference_inputs["input_text"]):
        raise ValueError("A model target has no usable input text.")
    return plan, inference_inputs, audit


def combine_news_type_predictions(
    plan: pd.DataFrame,
    predictions: pd.DataFrame,
    label2id: dict[str, int],
) -> pd.DataFrame:
    """Fill every unannotated target once, without replacing manual or hold rows."""
    _ordered_labels(label2id)
    if plan.columns.tolist() != OUTPUT_COLUMNS or predictions.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError("The inference plan or prediction table has an unexpected schema.")
    _validate_ids(plan, "Inference plan")
    _validate_ids(predictions, "Predictions")
    if not plan["label_source"].isin({"manual", "unlabeled", "review_required"}).all():
        raise ValueError("The inference plan contains an unknown or already-published label source.")
    manual = plan["label_source"].eq("manual")
    if not plan.loc[manual, "news_type"].isin(label2id).all():
        raise ValueError("Manual labels in the inference plan must be valid finalized labels.")
    if not all(_blank(value) for value in plan["prediction_score"]):
        raise ValueError("The inference plan must not invent scores for manual or pending labels.")
    if not all(_blank(value) for value in plan.loc[~manual, "news_type"]):
        raise ValueError("Unannotated and review-held articles must not contain a final label.")
    target_ids = plan.loc[plan["label_source"].eq("unlabeled"), "news_id"].tolist()
    if set(predictions["news_id"]) != set(target_ids):
        raise ValueError("Predictions must cover exactly the unannotated articles, never manual or hold rows.")
    if not predictions["news_type"].isin(label2id).all():
        raise ValueError("Predictions contain an unknown news_type.")
    if any(isinstance(value, (bool, np.bool_)) for value in predictions["prediction_score"]):
        raise ValueError("Prediction scores must be numeric Softmax scores, not booleans.")
    scores = pd.to_numeric(predictions["prediction_score"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Prediction scores must be finite and between zero and one.")
    result = plan.copy(deep=True)
    if target_ids:
        by_id = predictions.copy().set_index("news_id")
        by_id["prediction_score"] = scores
        targets = result["label_source"].eq("unlabeled")
        result.loc[targets, "news_type"] = by_id.loc[target_ids, "news_type"].to_numpy()
        result.loc[targets, "prediction_score"] = by_id.loc[target_ids, "prediction_score"].to_numpy()
        result.loc[targets, "label_source"] = "model"
    return result


class UnlabeledNewsDataset(Dataset):
    """Expose ID-aligned encoded articles without fake training labels."""

    def __init__(self, news_ids: Sequence[str], encodings: Sequence[Mapping[str, Any]]):
        if isinstance(news_ids, (str, bytes)) or isinstance(encodings, (str, bytes)) or len(news_ids) != len(encodings):
            raise ValueError("Article IDs and encodings must have the same length.")
        self.records: list[dict[str, torch.Tensor]] = []
        self.news_ids = list(news_ids)
        if any(not isinstance(news_id, str) or not news_id.strip() for news_id in news_ids) or len(set(news_ids)) != len(news_ids):
            raise ValueError("Encoded article IDs must be unique, nonempty strings.")
        sequence_length = None
        for news_id, encoding in zip(news_ids, encodings):
            if not isinstance(encoding, Mapping) or encoding.get("news_id") != news_id:
                raise ValueError("Encoding news_id order must match the model-target table.")
            if not {"input_ids", "attention_mask"}.issubset(encoding):
                raise ValueError("An encoding is missing input_ids or attention_mask.")
            input_ids = torch.as_tensor(encoding["input_ids"])
            attention_mask = torch.as_tensor(encoding["attention_mask"])
            if input_ids.ndim != 1 or input_ids.numel() == 0 or input_ids.dtype not in _INTEGER_DTYPES or bool((input_ids < 0).any()):
                raise ValueError("input_ids must be a nonempty one-dimensional nonnegative integer sequence.")
            if attention_mask.shape != input_ids.shape or not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask must match input_ids and contain only zero and one.")
            if not bool(attention_mask.any()) or bool((attention_mask[1:] > attention_mask[:-1]).any()):
                raise ValueError("Attention masks must contain real tokens followed only by right padding.")
            if sequence_length is None:
                sequence_length = input_ids.numel()
            elif sequence_length != input_ids.numel():
                raise ValueError("Every encoding must have the same padded sequence length.")
            self.records.append({
                "input_ids": input_ids.to(dtype=torch.long).clone(),
                "attention_mask": attention_mask.to(dtype=torch.long).clone(),
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.records[index]


def predict_news_types(
    model: nn.Module,
    prediction_data: pd.DataFrame,
    encodings: Sequence[Mapping[str, Any]],
    device: torch.device | str,
    label2id: dict[str, int],
    batch_size: int = 8,
) -> pd.DataFrame:
    """Run sequential evaluation only and return uncalibrated top-class scores."""
    ordered_labels = _ordered_labels(label2id)
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    _validate_ids(prediction_data, "Model targets")
    if "input_text" not in prediction_data or any(
        not isinstance(text, str) or not text.strip() for text in prediction_data["input_text"]
    ):
        raise ValueError("Every model target must contain nonempty input_text.")
    dataset = UnlabeledNewsDataset(prediction_data["news_id"].tolist(), encodings)
    if prediction_data.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    if not isinstance(model, nn.Module):
        raise TypeError("The inference model must be a PyTorch module.")
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.to(device)
    model.eval()
    records = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            logits = model(**{name: value.to(device) for name, value in batch.items()})
            if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch["input_ids"]), len(ordered_labels)):
                raise ValueError("Model logits must have shape [batch, number of labels].")
            if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
                raise ValueError("Model logits must contain finite floating-point values.")
            probabilities = torch.softmax(logits.float(), dim=1)
            if not bool(torch.isfinite(probabilities).all()):
                raise ValueError("Model Softmax scores are not finite.")
            scores, predicted_ids = probabilities.max(dim=1)
            for score, predicted_id in zip(scores.cpu().tolist(), predicted_ids.cpu().tolist()):
                records.append({
                    "news_id": dataset.news_ids[offset],
                    "news_type": ordered_labels[predicted_id],
                    "prediction_score": float(score),
                })
                offset += 1
    if offset != len(prediction_data):
        raise RuntimeError("Inference did not return exactly one prediction per model target.")
    return pd.DataFrame(records, columns=PREDICTION_COLUMNS)
