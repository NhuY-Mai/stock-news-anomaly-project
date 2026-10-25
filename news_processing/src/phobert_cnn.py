"""PhoBERT + TextCNN for news-type classification.

Four convolution branches capture local token patterns. A masked mean of the
backbone states supplies a complementary global representation. With the default
settings, the classifier receives 4 * 128 + 128 = 640 features.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


def _validate_encoding_options(tokenizer: Any, max_length: int, head_ratio: float) -> int:
    if isinstance(max_length, bool) or not isinstance(max_length, Integral):
        raise ValueError("max_length must be a positive integer.")
    if (
        isinstance(head_ratio, bool)
        or not isinstance(head_ratio, Real)
        or not math.isfinite(head_ratio)
        or not 0 <= head_ratio <= 1
    ):
        raise ValueError("head_ratio must be a finite number between 0 and 1.")

    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    if max_length <= special_token_count:
        raise ValueError("max_length must leave room for at least one content token.")

    model_max_length = getattr(tokenizer, "model_max_length", None)
    # Hugging Face uses a very large sentinel when no tokenizer limit is known.
    if isinstance(model_max_length, Integral) and 0 < model_max_length < 1_000_000:
        if max_length > model_max_length:
            raise ValueError(f"max_length exceeds the tokenizer limit ({model_max_length}).")
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define pad_token_id.")
    return int(max_length - special_token_count)


def encode_head_tail(
    text: str,
    tokenizer: Any,
    max_length: int = 256,
    head_ratio: float = 0.75,
) -> dict[str, Any]:
    """Encode segmented text, retaining its beginning and end when necessary.

    The token budget excludes special tokens. Right padding is applied only after
    token selection, so the attention mask identifies every real input position.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty segmented string.")
    content_budget = _validate_encoding_options(tokenizer, max_length, head_ratio)
    token_ids = list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
    if not token_ids:
        raise ValueError("The text produced no content tokens.")

    original_token_count = len(token_ids)
    if original_token_count > content_budget:
        head_count = int(content_budget * head_ratio)
        tail_count = content_budget - head_count
        head_tokens = token_ids[:head_count]
        tail_tokens = token_ids[-tail_count:] if tail_count else []
        token_ids = head_tokens + tail_tokens

    retained_token_count = len(token_ids)
    input_ids = list(tokenizer.build_inputs_with_special_tokens(token_ids))
    if len(input_ids) > max_length:
        raise ValueError("The tokenizer added more special tokens than declared.")
    attention_mask = [1] * len(input_ids)
    padding_count = max_length - len(input_ids)
    input_ids.extend([tokenizer.pad_token_id] * padding_count)
    attention_mask.extend([0] * padding_count)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "original_token_count": original_token_count,
        "retained_token_count": retained_token_count,
    }


def encode_news_texts(
    texts: Iterable[str],
    segmenter: Any,
    tokenizer: Any,
    max_length: int = 256,
    head_ratio: float = 0.75,
) -> list[dict[str, Any]]:
    """Segment Vietnamese sentences before applying PhoBERT's BPE tokenizer."""
    if isinstance(texts, str):
        raise ValueError("texts must be a collection of articles, not one string.")
    encodings = []
    for article_index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Article {article_index} has empty or invalid input_text.")
        sentences = segmenter.word_segment(text)
        if isinstance(sentences, str) or not isinstance(sentences, (list, tuple)):
            raise ValueError("segmenter.word_segment must return a list of sentences.")
        if any(not isinstance(sentence, str) for sentence in sentences):
            raise ValueError(f"The segmenter returned a non-string for article {article_index}.")
        segmented_text = " ".join(sentence.strip() for sentence in sentences).strip()
        if not segmented_text:
            raise ValueError(f"Article {article_index} is empty after word segmentation.")
        encodings.append(encode_head_tail(segmented_text, tokenizer, max_length, head_ratio))
    if not encodings:
        raise ValueError("texts must contain at least one article.")
    return encodings


class EncodedNewsDataset(Dataset):
    """Expose encoded articles and class IDs as tensors consumed by a DataLoader."""

    def __init__(self, encodings: Sequence[Mapping[str, Any]], labels: Sequence[int]):
        if len(encodings) != len(labels):
            raise ValueError("encodings and labels must have the same length.")
        self.records = []
        sequence_length = None
        for article_index, (encoding, label) in enumerate(zip(encodings, labels)):
            if isinstance(label, bool) or not isinstance(label, Integral) or label < 0:
                raise ValueError(f"Article {article_index} must have a non-negative integer label.")
            input_ids = torch.as_tensor(encoding["input_ids"], dtype=torch.long)
            attention_mask = torch.as_tensor(encoding["attention_mask"])
            if input_ids.ndim != 1 or input_ids.numel() == 0:
                raise ValueError("input_ids must be a non-empty one-dimensional sequence.")
            if attention_mask.shape != input_ids.shape:
                raise ValueError("input_ids and attention_mask must have the same shape.")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("attention_mask must contain only zero and one.")
            attention_mask = attention_mask.to(dtype=torch.long)
            if sequence_length is None:
                sequence_length = input_ids.numel()
            elif input_ids.numel() != sequence_length:
                raise ValueError("All articles must be padded to the same sequence length.")
            self.records.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(int(label), dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return self.records[index]


class PhoBertCNN(nn.Module):
    """Combine contextual CNN features and a projected masked mean representation.

    Padding is masked inside the backbone and before CNN max pooling. Special
    tokens remain valid positions in both pooling branches; only padding is
    excluded. The injected backbone makes the architecture independently testable.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 7,
        kernel_sizes: Sequence[int] = (2, 3, 4, 5),
        num_filters: int = 128,
        global_projection_size: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise ValueError("backbone must be a torch.nn.Module.")
        hidden_size = getattr(getattr(backbone, "config", None), "hidden_size", None)
        integer_options = {
            "hidden_size": hidden_size,
            "num_classes": num_classes,
            "num_filters": num_filters,
            "global_projection_size": global_projection_size,
        }
        for option_name, option_value in integer_options.items():
            if isinstance(option_value, bool) or not isinstance(option_value, Integral) or option_value < 1:
                raise ValueError(f"{option_name} must be a positive integer.")
        self.kernel_sizes = tuple(kernel_sizes)
        if not self.kernel_sizes or any(
            isinstance(kernel_size, bool) or not isinstance(kernel_size, Integral) or kernel_size < 1
            for kernel_size in self.kernel_sizes
        ):
            raise ValueError("kernel_sizes must contain positive integers.")
        if len(set(self.kernel_sizes)) != len(self.kernel_sizes):
            raise ValueError("kernel_sizes must not contain duplicates.")
        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not 0 <= dropout < 1:
            raise ValueError("dropout must be a number in [0, 1).")

        self.backbone = backbone
        self.convolutions = nn.ModuleList([
            nn.Conv1d(hidden_size, num_filters, kernel_size)
            for kernel_size in self.kernel_sizes
        ])
        self.global_projection = nn.Linear(hidden_size, global_projection_size)
        self.feature_size = len(self.kernel_sizes) * num_filters + global_projection_size
        self.layer_norm = nn.LayerNorm(self.feature_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_size, num_classes)

    @staticmethod
    def _pool_convolution(
        token_features: Tensor,
        valid_tokens: Tensor,
        convolution: nn.Conv1d,
        kernel_size: int,
    ) -> Tensor:
        if token_features.shape[-1] < kernel_size:
            return token_features.new_zeros((token_features.shape[0], convolution.out_channels))

        activations = F.gelu(convolution(token_features))
        valid_windows = valid_tokens.unfold(1, kernel_size, 1).all(dim=-1)
        masked_activations = activations.masked_fill(~valid_windows.unsqueeze(1), -torch.inf)
        pooled_features = masked_activations.amax(dim=-1)
        return torch.where(valid_windows.any(dim=-1, keepdim=True), pooled_features, 0.0)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have non-empty shape [batch, sequence_length].")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only zero and one.")

        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        valid_tokens = attention_mask.bool()
        # Some attention implementations can emit NaNs for fully padded inputs.
        hidden_states = hidden_states.masked_fill(~valid_tokens.unsqueeze(-1), 0.0)
        token_features = hidden_states.transpose(1, 2)
        cnn_features = [
            self._pool_convolution(token_features, valid_tokens, convolution, kernel_size)
            for convolution, kernel_size in zip(self.convolutions, self.kernel_sizes)
        ]

        token_count = valid_tokens.sum(dim=1, keepdim=True).clamp_min(1)
        global_mean = hidden_states.sum(dim=1) / token_count.to(hidden_states.dtype)
        global_features = self.global_projection(global_mean)
        all_features = torch.cat([*cnn_features, global_features], dim=1)
        normalized_features = self.layer_norm(all_features)
        return self.classifier(self.dropout(normalized_features))
