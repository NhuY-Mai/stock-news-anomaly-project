"""Offline checks for token selection, padding masks and the training loop."""

import hashlib
import json
import os
import shutil
import sys
import weakref
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ImportError:
    pytest.skip("PyTorch is required for model tests.", allow_module_level=True)

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import src.news_type_training as training_helpers
from src.phobert_cnn import (
    EncodedNewsDataset,
    PhoBertCNN,
    encode_head_tail,
    encode_news_texts,
)
from src.news_type_training import (
    evaluate_classifier,
    fit_classifier,
    make_optimizer,
    make_scheduler,
    rank_parameter_trials,
    run_parameter_study,
    seed_everything,
    train_one_epoch,
)


class FakeTokenizer:
    """Use transparent token IDs instead of downloading a pretrained tokenizer."""

    pad_token_id = 1
    model_max_length = 256

    def __init__(self, events=None):
        self.events = events if events is not None else []

    def encode(self, text, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False
        assert truncation is False
        self.events.append(("tokenize", text))
        return [10 + int(token.removeprefix("t")) for token in text.split()]

    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return 2

    def build_inputs_with_special_tokens(self, token_ids):
        return [0, *token_ids, 2]


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(64, hidden_size)
        self.last_attention_mask = None

    def forward(self, input_ids, attention_mask, return_dict=True):
        assert return_dict is True
        self.last_attention_mask = attention_mask.detach().clone()
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyClassificationDataset(Dataset):
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([index + 3, index + 9], dtype=torch.long),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Embedding(32, 5)
        self.classifier = nn.Linear(5, 3)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.backbone(input_ids)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(pooled)


class FixedLogitClassifier(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits.clone())

    def forward(self, input_ids, attention_mask):
        return self.logits[input_ids[:, 0] - 3]


def test_head_tail_reserves_special_tokens_and_preserves_both_ends():
    text = " ".join(f"t{index}" for index in range(10))
    encoding = encode_head_tail(text, FakeTokenizer(), max_length=8, head_ratio=0.5)

    assert encoding["input_ids"] == [0, 10, 11, 12, 17, 18, 19, 2]
    assert encoding["attention_mask"] == [1] * 8
    assert encoding["original_token_count"] == 10
    assert encoding["retained_token_count"] == 6


@pytest.mark.parametrize(
    ("head_ratio", "expected_content"),
    [(1.0, [10, 11, 12, 13, 14, 15]), (0.0, [14, 15, 16, 17, 18, 19])],
)
def test_head_tail_endpoint_ratios_do_not_duplicate_the_whole_article(
    head_ratio, expected_content
):
    text = " ".join(f"t{index}" for index in range(10))
    encoding = encode_head_tail(text, FakeTokenizer(), max_length=8, head_ratio=head_ratio)
    assert encoding["input_ids"] == [0, *expected_content, 2]


def test_short_article_is_not_duplicated_and_is_right_padded():
    encoding = encode_head_tail("t0 t1 t2", FakeTokenizer(), max_length=8)
    assert encoding["input_ids"] == [0, 10, 11, 12, 2, 1, 1, 1]
    assert encoding["attention_mask"] == [1, 1, 1, 1, 1, 0, 0, 0]
    assert encoding["original_token_count"] == encoding["retained_token_count"] == 3


def test_default_input_budget_is_254_content_tokens():
    text = " ".join(f"t{index}" for index in range(300))
    encoding = encode_head_tail(text, FakeTokenizer())
    assert len(encoding["input_ids"]) == 256
    assert encoding["retained_token_count"] == 254
    assert encoding["original_token_count"] == 300
    assert encoding["input_ids"][0] == 0
    assert encoding["input_ids"][-1] == 2


@pytest.mark.parametrize("max_length", [0, 2, 257, 8.5, True])
def test_head_tail_rejects_invalid_lengths(max_length):
    with pytest.raises(ValueError):
        encode_head_tail("t0", FakeTokenizer(), max_length=max_length)


@pytest.mark.parametrize("head_ratio", [-0.01, 1.01, float("nan"), True])
def test_head_tail_rejects_invalid_ratios(head_ratio):
    with pytest.raises(ValueError):
        encode_head_tail("t0", FakeTokenizer(), head_ratio=head_ratio)


@pytest.mark.parametrize("text", ["", "   ", None])
def test_head_tail_rejects_empty_or_invalid_text(text):
    with pytest.raises(ValueError):
        encode_head_tail(text, FakeTokenizer())


def test_word_segmentation_precedes_tokenization_for_each_article():
    events = []

    class FakeSegmenter:
        def word_segment(self, text):
            events.append(("segment", text))
            return {"first article": ["t0 t1", "t2"], "second article": ["t3"]}[text]

    encodings = encode_news_texts(
        ["first article", "second article"], FakeSegmenter(), FakeTokenizer(events), 8, 0.75
    )
    assert events == [
        ("segment", "first article"),
        ("tokenize", "t0 t1 t2"),
        ("segment", "second article"),
        ("tokenize", "t3"),
    ]
    assert len(encodings) == 2


def test_encoded_dataset_preserves_label_alignment_and_tensor_types():
    encodings = [
        encode_head_tail("t0 t1", FakeTokenizer(), max_length=8),
        encode_head_tail("t2 t3", FakeTokenizer(), max_length=8),
    ]
    dataset = EncodedNewsDataset(encodings, [6, 2])
    assert len(dataset) == 2
    assert dataset[0]["labels"].item() == 6
    assert dataset[1]["labels"].item() == 2
    assert set(dataset[0]) == {"input_ids", "attention_mask", "labels"}
    assert all(value.dtype == torch.long for value in dataset[0].values())
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].tolist() == [6, 2]


def test_encoded_dataset_rejects_misaligned_labels_or_masks():
    encoding = encode_head_tail("t0", FakeTokenizer(), max_length=8)
    with pytest.raises(ValueError):
        EncodedNewsDataset([encoding], [])
    with pytest.raises(ValueError):
        EncodedNewsDataset([encoding], [-1])
    with pytest.raises(ValueError):
        EncodedNewsDataset([{**encoding, "attention_mask": [1]}], [0])


def test_phobert_cnn_default_features_and_output_shape():
    backbone = TinyBackbone()
    model = PhoBertCNN(backbone).eval()
    input_ids = torch.randint(3, 32, (3, 12))
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids, mask)
    assert model.feature_size == 640
    assert logits.shape == (3, 7)
    assert torch.isfinite(logits).all()
    assert torch.equal(backbone.last_attention_mask, mask)


def test_padding_ids_and_additional_padding_do_not_change_predictions():
    torch.manual_seed(4)
    model = PhoBertCNN(TinyBackbone(), dropout=0.0).eval()
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    original = torch.tensor([[3, 4, 5, 6, 7, 1, 1]])
    perturbed = torch.tensor([[3, 4, 5, 6, 7, 51, 52]])
    longer = torch.tensor([[3, 4, 5, 6, 7, 51, 52, 53, 54, 55]])
    longer_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

    with torch.no_grad():
        expected = model(original, mask)
        changed_ids = model(perturbed, mask)
        changed_length = model(longer, longer_mask)
    torch.testing.assert_close(changed_ids, expected)
    torch.testing.assert_close(changed_length, expected)


def test_cnn_masks_invalid_windows_after_activation_despite_positive_bias():
    convolution = nn.Conv1d(1, 1, kernel_size=2)
    with torch.no_grad():
        convolution.weight.fill_(1.0)
        convolution.bias.fill_(1.0)
    token_features = torch.tensor([[[-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0]]])
    valid_tokens = torch.tensor([[True, True, True, True, True, False, False]])

    pooled = PhoBertCNN._pool_convolution(token_features, valid_tokens, convolution, 2)
    # Invalid padding windows have GELU(1) > GELU(-1), so zeroing inputs alone fails.
    torch.testing.assert_close(pooled, F.gelu(torch.tensor([[-1.0]])))


@pytest.mark.parametrize("sequence_length", [1, 2, 4, 7])
def test_short_and_fully_padded_inputs_have_finite_outputs(sequence_length):
    model = PhoBertCNN(TinyBackbone(), dropout=0.0).eval()
    input_ids = torch.full((2, sequence_length), 3, dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    mask[0, 0] = 1
    with torch.no_grad():
        logits = model(input_ids, mask)
    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()


def test_gradients_reach_backbone_convolutions_and_global_projection():
    torch.manual_seed(42)
    model = PhoBertCNN(TinyBackbone(), num_classes=3, dropout=0.0)
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14]])
    loss = F.cross_entropy(model(input_ids, torch.ones_like(input_ids)), torch.tensor([0, 2]))
    loss.backward()

    assert model.backbone.embedding.weight.grad.abs().sum() > 0
    assert all(layer.weight.grad.abs().sum() > 0 for layer in model.convolutions)
    assert model.global_projection.weight.grad.abs().sum() > 0
    assert model.classifier.weight.grad.abs().sum() > 0


def test_seeding_reproduces_model_initialization():
    seed_everything(42)
    first = TinyClassifier()
    seed_everything(42)
    second = TinyClassifier()
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_optimizer_assigns_backbone_and_head_rates_without_duplicate_parameters():
    model = TinyClassifier()
    optimizer = make_optimizer(model, backbone_lr=2e-5, head_lr=1e-3, weight_decay=0.01)
    parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert set(parameter_ids) == expected_ids
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    for group in optimizer.param_groups:
        expected_rate = 2e-5 if id(group["params"][0]) in backbone_ids else 1e-3
        assert group["lr"] == expected_rate


def test_scheduler_has_finite_nonnegative_learning_rates_and_decays():
    model = TinyClassifier()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, total_steps=10, warmup_ratio=0.2)
    head_rates = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        head_rates.append(max(group["lr"] for group in optimizer.param_groups))
    assert all(np.isfinite(rate) and rate >= 0 for rate in head_rates)
    assert max(head_rates) > head_rates[-1]


def test_evaluation_weighted_loss_is_computed_over_the_full_dataset():
    logits = torch.tensor([
        [3.0, 0.0, -1.0],
        [0.0, 1.0, 2.0],
        [-1.0, 2.0, 0.0],
        [2.0, 0.0, 1.0],
        [0.0, -1.0, 3.0],
    ])
    labels = [0, 0, 1, 2, 2]
    model = FixedLogitClassifier(logits)
    loader = DataLoader(TinyClassificationDataset(labels), batch_size=3, shuffle=False)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0, 9.0]))

    result = evaluate_classifier(model, loader, "cpu", num_classes=3, criterion=criterion)
    expected_loss = criterion(logits, torch.tensor(labels)).item()
    assert result["loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert result["labels"] == labels
    assert result["predictions"] == logits.argmax(1).tolist()
    assert result["probabilities"].shape == (5, 3)
    np.testing.assert_allclose(result["probabilities"].sum(1), 1.0, atol=1e-6)
    assert result["accuracy"] == pytest.approx(3 / 5)


def test_one_epoch_updates_parameters_and_returns_finite_metrics():
    seed_everything(42)
    model = TinyClassifier()
    loader = DataLoader(TinyClassificationDataset([0, 1, 2, 0, 1, 2]), batch_size=2)
    optimizer = make_optimizer(model, backbone_lr=1e-3, head_lr=1e-2)
    scheduler = make_scheduler(optimizer, total_steps=len(loader), warmup_ratio=0.0)
    original_weights = model.backbone.weight.detach().clone()

    result = train_one_epoch(
        model, loader, optimizer, scheduler, nn.CrossEntropyLoss(), "cpu", num_classes=3
    )
    assert not torch.equal(model.backbone.weight, original_weights)
    for metric_name in ("loss", "accuracy", "macro_f1"):
        assert np.isfinite(result[metric_name])


def test_training_weighted_loss_does_not_average_batch_means():
    logits = torch.tensor([
        [3.0, 0.0, -1.0], [0.0, 1.0, 2.0], [-1.0, 2.0, 0.0],
        [2.0, 0.0, 1.0], [0.0, -1.0, 3.0],
    ])
    labels = [0, 0, 1, 2, 2]
    model = FixedLogitClassifier(logits)
    loader = DataLoader(TinyClassificationDataset(labels), batch_size=3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0, 9.0]))
    # Keep predictions fixed so the expected whole-epoch weighted loss is exact.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    result = train_one_epoch(model, loader, optimizer, None, criterion, "cpu", 3)
    assert result["loss"] == pytest.approx(
        criterion(logits, torch.tensor(labels)).item(), abs=1e-6
    )


def test_fit_saves_a_reloadable_validation_selected_checkpoint(tmp_path):
    seed_everything(42)
    model = TinyClassifier()
    train_loader = DataLoader(TinyClassificationDataset([0, 1, 2, 0, 1, 2]), batch_size=3)
    validation_loader = DataLoader(TinyClassificationDataset([0, 1, 2]), batch_size=3)
    optimizer = make_optimizer(model, backbone_lr=1e-3, head_lr=1e-2)
    scheduler = make_scheduler(optimizer, total_steps=len(train_loader), warmup_ratio=0.0)
    checkpoint_path = tmp_path / "best_model.pt"

    history = fit_classifier(
        model, train_loader, validation_loader, optimizer, scheduler,
        nn.CrossEntropyLoss(), "cpu", num_classes=3,
        checkpoint_path=checkpoint_path, max_epochs=1, patience=2,
    )

    assert len(history) == 1
    assert bool(history.iloc[0]["is_best"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["validation_macro_f1"] == pytest.approx(history.iloc[0]["validation_macro_f1"])
    assert all(value.device.type == "cpu" for value in checkpoint["model_state_dict"].values())

    restored_model = TinyClassifier()
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_result = evaluate_classifier(restored_model, validation_loader, "cpu", num_classes=3)
    assert restored_result["macro_f1"] == pytest.approx(checkpoint["validation_macro_f1"])

    with pytest.raises(FileExistsError):
        fit_classifier(
            model, train_loader, validation_loader, optimizer, scheduler,
            nn.CrossEntropyLoss(), "cpu", num_classes=3,
            checkpoint_path=checkpoint_path, max_epochs=1,
        )


def test_early_stopping_restores_highest_validation_score_not_last_epoch(tmp_path, monkeypatch):
    model = TinyClassifier()
    train_loader, validation_loader = object(), object()
    validation_scores = iter([0.4, 0.40005, 0.1])
    epoch_number = 0

    def fake_train(current_model, loader, *args, **kwargs):
        nonlocal epoch_number
        assert loader is train_loader
        epoch_number += 1
        with torch.no_grad():
            current_model.classifier.bias.fill_(float(epoch_number))
        return {"loss": 1.0, "accuracy": 0.5, "macro_f1": 0.5}

    def fake_evaluate(current_model, loader, *args, **kwargs):
        assert loader is validation_loader
        return {"loss": 1.0, "accuracy": 0.5, "macro_f1": next(validation_scores)}

    monkeypatch.setattr(training_helpers, "train_one_epoch", fake_train)
    monkeypatch.setattr(training_helpers, "evaluate_classifier", fake_evaluate)
    optimizer = make_optimizer(model)
    history = fit_classifier(
        model, train_loader, validation_loader, optimizer, None,
        nn.CrossEntropyLoss(), "cpu", 3, tmp_path / "best.pt",
        max_epochs=8, patience=2, min_delta=1e-4,
    )

    assert len(history) == 3
    assert history.attrs["best_epoch"] == 2
    assert history.attrs["best_validation_macro_f1"] == pytest.approx(0.40005)
    torch.testing.assert_close(model.classifier.bias, torch.full((3,), 2.0))
    assert model.training is False


def _notebook_cell_source(cell_id):
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks/phase_1_news_type/05_train_phobert_cnn.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)
    return "".join(cell["source"])


def test_notebook_defaults_to_one_fresh_run_without_study_or_test():
    namespace = {}

    exec(_notebook_cell_source("news05-07"), namespace)

    assert namespace["DROPOUT"] == 0.5
    assert namespace["RUN_PARAMETER_STUDY"] is False
    assert namespace["EVALUATE_TEST"] is False


@pytest.mark.parametrize(
    "policy_version, accepted",
    [(None, False), ("gold_price_industry_v0", False), ("gold_price_industry_v1", False),
     ("company_awards_listing_v2", True)],
)
def test_notebook_requires_current_label_policy_before_reading_splits(
    tmp_path, policy_version, accepted,
):
    mapping = {"label2id": {"industry": 0}}
    if policy_version is not None:
        mapping["label_policy_version"] = policy_version
    mapping_path = tmp_path / "label_mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for split_name in ("train", "validation", "test"):
        (split_dir / f"{split_name}.csv").write_text("news_id\n", encoding="utf-8")
    read_calls = []

    def read_split(path, **kwargs):
        read_calls.append(path.name)
        return pd.DataFrame(columns=["news_id"])

    namespace = {
        "LABEL_MAPPING_PATH": mapping_path, "SPLIT_DIR": split_dir,
        "PROJECT_DIR": tmp_path, "json": json, "hashlib": hashlib,
        "pd": SimpleNamespace(read_csv=read_split),
    }
    if accepted:
        exec(_notebook_cell_source("news05-11"), namespace)
        assert read_calls == ["train.csv", "validation.csv", "test.csv"]
        assert namespace["label_mapping"] == mapping
        assert len(namespace["input_file_hashes"]) == 4
    else:
        with pytest.raises(ValueError):
            exec(_notebook_cell_source("news05-11"), namespace)
        assert read_calls == []


def _create_fake_java_home(java_home):
    java_bin = java_home / "bin"
    java_bin.mkdir(parents=True)
    (java_bin / ("java.exe" if os.name == "nt" else "java")).touch()
    return java_home


def test_notebook_cpu_preflight_warns_but_allows_java_ready_execution(monkeypatch, capsys):
    monkeypatch.setattr(shutil, "which", lambda command: "fake-java" if command == "java" else None)
    namespace = {"DEVICE": SimpleNamespace(type="cpu"), "shutil": shutil}

    exec(_notebook_cell_source("news05-16"), namespace)

    output = capsys.readouterr().out
    assert "CPU" in output
    assert "Java" in output


def test_notebook_preflight_reports_missing_java_without_starting_training(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda command: None)
    namespace = {"DEVICE": SimpleNamespace(type="cpu"), "shutil": shutil}

    with pytest.raises(RuntimeError, match="Java.*models/java"):
        exec(_notebook_cell_source("news05-16"), namespace)


def test_notebook_java_preflight_uses_portable_runtime_without_global_installation(
    tmp_path, monkeypatch
):
    java_home = _create_fake_java_home(tmp_path / "models/java/portable-jre")
    for variable_name in ("JAVA_HOME", "JDK_HOME", "JRE_HOME"):
        monkeypatch.delenv(variable_name, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "original-bin"))
    monkeypatch.setattr(shutil, "which", lambda command: None)
    namespace = {"PROJECT_DIR": tmp_path, "Path": Path, "os": os, "shutil": shutil}

    exec(_notebook_cell_source("news05-java-setup"), namespace)

    assert os.environ["JAVA_HOME"] == str(java_home.resolve())
    assert os.environ["PATH"].split(os.pathsep) == [
        str(java_home.resolve() / "bin"), str(tmp_path / "original-bin")
    ]


def test_notebook_java_preflight_prefers_existing_runtime_and_deduplicates_path(
    tmp_path, monkeypatch
):
    existing_java_home = _create_fake_java_home(tmp_path / "existing-jre")
    portable_java_home = _create_fake_java_home(tmp_path / "models/java/portable-jre")
    monkeypatch.setenv("JAVA_HOME", str(existing_java_home))
    monkeypatch.delenv("JDK_HOME", raising=False)
    monkeypatch.delenv("JRE_HOME", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "original-bin"))
    monkeypatch.setattr(shutil, "which", lambda command: None)
    namespace = {"PROJECT_DIR": tmp_path, "Path": Path, "os": os, "shutil": shutil}
    source = _notebook_cell_source("news05-java-setup")

    exec(source, namespace)
    first_path = os.environ["PATH"]
    exec(source, namespace)

    assert os.environ["JAVA_HOME"] == str(existing_java_home.resolve())
    assert os.environ["PATH"] == first_path
    path_entries = os.environ["PATH"].split(os.pathsep)
    assert path_entries.count(str(existing_java_home.resolve() / "bin")) == 1
    assert str(portable_java_home.resolve() / "bin") not in path_entries


def _mock_colab_import(monkeypatch, available=True):
    google_module = ModuleType("google")
    google_module.__path__ = []
    colab_module = ModuleType("google.colab") if available else None
    google_module.colab = colab_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)
    return colab_module


def _mock_torch_runtime(monkeypatch, gpu_available=True, version="2.9.0+cu128"):
    runtime = ModuleType("torch")
    runtime.__version__ = version
    runtime.cuda = SimpleNamespace(
        is_available=lambda: gpu_available,
        get_device_name=lambda index=0: "Fake Colab GPU",
        get_device_properties=lambda index=0: SimpleNamespace(total_memory=16 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", runtime)
    return runtime


def _remap_colab_paths(content_root):
    def mapped_path(*parts):
        if parts and str(parts[0]).replace("\\", "/").startswith("/content"):
            relative_path = str(parts[0]).replace("\\", "/").removeprefix("/content").lstrip("/")
            return Path(content_root, relative_path, *parts[1:])
        return Path(*parts)

    mapped_path.cwd = Path.cwd
    return mapped_path


@pytest.mark.parametrize(
    ("gpu_available", "torch_version"), [(False, "2.9.0+cu128"), (True, "2.5.1+cu124")]
)
def test_colab_runtime_rejects_missing_gpu_or_unsupported_torch(
    monkeypatch, gpu_available, torch_version
):
    _mock_colab_import(monkeypatch)
    runtime = _mock_torch_runtime(monkeypatch, gpu_available, torch_version)
    with pytest.raises(RuntimeError):
        exec(_notebook_cell_source("news05-colab-runtime"), {"torch": runtime})


def test_colab_runtime_accepts_available_gpu_without_replacing_torch(monkeypatch, capsys):
    _mock_colab_import(monkeypatch)
    runtime = _mock_torch_runtime(monkeypatch)
    namespace = {"torch": runtime}

    exec(_notebook_cell_source("news05-colab-runtime"), namespace)

    assert namespace["IN_COLAB"] is True
    assert sys.modules["torch"] is runtime
    assert "GPU" in capsys.readouterr().out


def test_colab_runtime_does_not_require_gpu_on_local_cpu(monkeypatch):
    _mock_colab_import(monkeypatch, available=False)
    runtime = _mock_torch_runtime(monkeypatch, gpu_available=False)
    namespace = {"torch": runtime}

    exec(_notebook_cell_source("news05-colab-runtime"), namespace)

    assert namespace["IN_COLAB"] is False


@pytest.mark.parametrize("in_colab", [False, True])
def test_colab_drive_mount_only_runs_inside_colab(tmp_path, monkeypatch, in_colab):
    mount_calls = []
    colab_module = _mock_colab_import(monkeypatch)
    colab_module.drive = SimpleNamespace(
        mount=lambda *args, **kwargs: mount_calls.append((args, kwargs))
    )
    namespace = {"IN_COLAB": in_colab, "Path": _remap_colab_paths(tmp_path)}

    exec(_notebook_cell_source("news05-colab-drive"), namespace)

    assert len(mount_calls) == int(in_colab)
    if in_colab:
        assert str(namespace["COLAB_OUTPUT_DIR"]).endswith("stock_news_anomaly_project")


@pytest.mark.parametrize(
    ("in_colab", "java_available"), [(False, False), (True, True), (True, False)]
)
def test_colab_dependency_setup_keeps_cuda_torch_and_installs_java_only_if_missing(
    monkeypatch, in_colab, java_available
):
    command_calls = []
    monkeypatch.setattr(shutil, "which", lambda command: "fake-java" if java_available else None)
    namespace = {
        "IN_COLAB": in_colab,
        "sys": sys,
        "shutil": shutil,
        "subprocess": SimpleNamespace(check_call=lambda command: command_calls.append(command)),
    }

    exec(_notebook_cell_source("news05-colab-deps"), namespace)

    if not in_colab:
        assert command_calls == []
        return
    pip_command = command_calls[0]
    assert pip_command[:4] == [sys.executable, "-m", "pip", "install"]
    assert "transformers==4.57.1" in pip_command
    assert "py-vncorenlp==0.1.4" in pip_command
    assert not any(argument.startswith(("torch", "pytorch")) for argument in pip_command)
    apt_commands = [command for command in command_calls if command[0] == "apt-get"]
    assert len(apt_commands) == (0 if java_available else 2)


def _colab_archive_namespace(tmp_path):
    mapped_path = _remap_colab_paths(tmp_path)
    extraction_calls = []

    class MappedZipFile(zipfile.ZipFile):
        def extractall(self, path=None, *args, **kwargs):
            extraction_calls.append(path)
            return super().extractall(mapped_path(path), *args, **kwargs)

    namespace = {
        "IN_COLAB": True,
        "Path": mapped_path,
        "zipfile": SimpleNamespace(ZipFile=MappedZipFile),
        "COLAB_OUTPUT_DIR": tmp_path / "drive-output",
    }
    return namespace, extraction_calls


def test_colab_project_archive_extracts_only_the_expected_project(tmp_path):
    with zipfile.ZipFile(tmp_path / "news_processing_colab.zip", "w") as archive:
        archive.writestr("news_processing/src/example.py", "# Offline fixture\n")
        archive.writestr("news_processing/data/splits/news_type/train.csv", "news_id\n")
    namespace, extraction_calls = _colab_archive_namespace(tmp_path)

    exec(_notebook_cell_source("news05-colab-project"), namespace)

    assert extraction_calls == ["/content"]
    assert namespace["COLAB_PROJECT_DIR"] == tmp_path / "news_processing"
    assert (tmp_path / "news_processing/src/example.py").is_file()
    assert (tmp_path / "news_processing/data/splits/news_type/train.csv").is_file()


@pytest.mark.parametrize(
    "member_name",
    ["../notebook5-escaped.txt", "news_processing/../../notebook5-escaped.txt", "other_project/data.txt"],
)
def test_colab_project_archive_rejects_paths_outside_the_project(tmp_path, member_name):
    with zipfile.ZipFile(tmp_path / "news_processing_colab.zip", "w") as archive:
        archive.writestr(member_name, "must not be extracted")
    namespace, extraction_calls = _colab_archive_namespace(tmp_path)

    with pytest.raises(ValueError):
        exec(_notebook_cell_source("news05-colab-project"), namespace)

    assert extraction_calls == []
    assert not (tmp_path.parent / "notebook5-escaped.txt").exists()


def test_colab_existing_project_does_not_require_a_second_upload(tmp_path):
    (tmp_path / "news_processing/src").mkdir(parents=True)
    (tmp_path / "news_processing/data").mkdir()
    namespace, extraction_calls = _colab_archive_namespace(tmp_path)

    exec(_notebook_cell_source("news05-colab-project"), namespace)

    assert extraction_calls == []
    assert not (tmp_path / "news_processing_colab.zip").exists()


def test_colab_uploaded_archive_refreshes_an_existing_project(tmp_path):
    source_dir = tmp_path / "news_processing/src"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "example.py"
    source_path.write_text("# Previous source\n", encoding="utf-8")
    cache_path = source_dir / "existing-cache.txt"
    cache_path.write_text("preserve", encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "news_processing_colab.zip", "w") as archive:
        archive.writestr("news_processing/src/example.py", "# Updated source\n")
        archive.writestr("news_processing/data/splits/news_type/train.csv", "news_id\n")
    namespace, extraction_calls = _colab_archive_namespace(tmp_path)

    exec(_notebook_cell_source("news05-colab-project"), namespace)

    assert extraction_calls == ["/content"]
    assert source_path.read_text(encoding="utf-8") == "# Updated source\n"
    assert cache_path.read_text(encoding="utf-8") == "preserve"
    assert (tmp_path / "news_processing/data/splits/news_type/train.csv").is_file()


def test_colab_drive_archive_takes_priority_over_stale_content_archive(tmp_path):
    namespace, extraction_calls = _colab_archive_namespace(tmp_path)
    drive_archive = namespace["COLAB_OUTPUT_DIR"] / "news_processing_colab.zip"
    drive_archive.parent.mkdir(parents=True)
    for archive_path, source in (
        (tmp_path / "news_processing_colab.zip", "# Stale content archive\n"),
        (drive_archive, "# Current Drive archive\n"),
    ):
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("news_processing/src/example.py", source)

    exec(_notebook_cell_source("news05-colab-project"), namespace)

    assert extraction_calls == ["/content"]
    assert namespace["project_archive"] == drive_archive
    assert (tmp_path / "news_processing/src/example.py").read_text() == "# Current Drive archive\n"


def test_colab_drive_archive_bootstraps_without_a_content_upload(tmp_path):
    namespace, extraction_calls = _colab_archive_namespace(tmp_path)
    drive_archive = namespace["COLAB_OUTPUT_DIR"] / "news_processing_colab.zip"
    drive_archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(drive_archive, "w") as archive:
        archive.writestr("news_processing/src/example.py", "# Drive-only fixture\n")
        archive.writestr("news_processing/data/splits/news_type/train.csv", "news_id\n")
    assert not (tmp_path / "news_processing_colab.zip").exists()
    assert not (tmp_path / "news_processing").exists()

    exec(_notebook_cell_source("news05-colab-project"), namespace)

    assert extraction_calls == ["/content"]
    assert namespace["project_archive"] == drive_archive
    assert (tmp_path / "news_processing/src/example.py").is_file()
    assert (tmp_path / "news_processing/data/splits/news_type/train.csv").is_file()


def test_colab_unsafe_drive_archive_never_extracts_or_falls_back_to_content(tmp_path):
    namespace, extraction_calls = _colab_archive_namespace(tmp_path)
    source_path = tmp_path / "news_processing/src/example.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# Existing source\n", encoding="utf-8")
    drive_archive = namespace["COLAB_OUTPUT_DIR"] / "news_processing_colab.zip"
    drive_archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(drive_archive, "w") as archive:
        archive.writestr("news_processing/src/example.py", "must not overwrite existing source")
        archive.writestr("../notebook5-drive-escaped.txt", "must not escape")
    with zipfile.ZipFile(tmp_path / "news_processing_colab.zip", "w") as archive:
        archive.writestr("news_processing/src/example.py", "must not use fallback archive")
        archive.writestr("news_processing/src/fallback-only.py", "must not create fallback file")

    with pytest.raises(ValueError):
        exec(_notebook_cell_source("news05-colab-project"), namespace)

    assert extraction_calls == []
    assert namespace["project_archive"] == drive_archive
    assert source_path.read_text(encoding="utf-8") == "# Existing source\n"
    assert not source_path.with_name("fallback-only.py").exists()
    assert not (tmp_path.parent / "notebook5-drive-escaped.txt").exists()


@pytest.mark.parametrize(
    "member_name",
    ["../notebook5-escaped.txt", "news_processing/../../notebook5-escaped.txt", "other_project/data.txt"],
)
def test_colab_existing_project_rejects_unsafe_archive_before_any_extraction(
    tmp_path, member_name,
):
    source_dir = tmp_path / "news_processing/src"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "example.py"
    source_path.write_text("# Previous source\n", encoding="utf-8")
    with zipfile.ZipFile(tmp_path / "news_processing_colab.zip", "w") as archive:
        archive.writestr("news_processing/src/example.py", "must not overwrite")
        archive.writestr(member_name, "must not escape")
    namespace, extraction_calls = _colab_archive_namespace(tmp_path)

    with pytest.raises(ValueError):
        exec(_notebook_cell_source("news05-colab-project"), namespace)

    assert extraction_calls == []
    assert source_path.read_text(encoding="utf-8") == "# Previous source\n"
    assert not (tmp_path.parent / "notebook5-escaped.txt").exists()
    assert not (tmp_path / "other_project").exists()


def test_colab_project_finder_prefers_uploaded_project(tmp_path, monkeypatch):
    uploaded_project = tmp_path / "uploaded-news-processing"
    (uploaded_project / "src").mkdir(parents=True)
    (uploaded_project / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    namespace = {
        "IN_COLAB": True,
        "COLAB_PROJECT_DIR": uploaded_project,
        "Path": Path,
        "sys": SimpleNamespace(path=list(sys.path)),
    }

    exec(_notebook_cell_source("news05-04"), namespace)

    assert namespace["PROJECT_DIR"] == uploaded_project
    assert namespace["sys"].path[0] == str(uploaded_project)


@pytest.mark.parametrize("in_colab", [False, True])
def test_colab_artifacts_use_drive_but_labels_and_caches_remain_in_the_project(tmp_path, in_colab):
    project_dir = tmp_path / "project"
    drive_dir = tmp_path / "drive-output"
    namespace = {
        "IN_COLAB": in_colab,
        "PROJECT_DIR": project_dir,
        "COLAB_OUTPUT_DIR": drive_dir if in_colab else None,
    }

    exec(_notebook_cell_source("news05-06"), namespace)

    output_root = drive_dir if in_colab else project_dir
    assert namespace["MODEL_DIR"] == output_root / "models/news_type_phobert_cnn"
    assert namespace["REPORT_DIR"] == output_root / "reports/news_type"
    assert namespace["LABEL_MAPPING_PATH"] == project_dir / "models/news_type_phobert_cnn/label_mapping.json"
    assert namespace["MODEL_CACHE_DIR"] == project_dir / "models/huggingface"
    assert namespace["VNCORENLP_DIR"] == project_dir / "models/vncorenlp"
    assert namespace["SPLIT_DIR"] == project_dir / "data/splits/news_type"


def _study_configs():
    return [
        {"trial_id": "baseline", "backbone_lr": 1e-3, "head_lr": 1e-2, "dropout": 0.3, "weight_decay": 0.01},
        {"trial_id": "lower_dropout", "backbone_lr": 1e-3, "head_lr": 1e-2, "dropout": 0.1, "weight_decay": 0.01},
    ]


def _study_loaders(config):
    train_loader = DataLoader(TinyClassificationDataset([0, 1, 2, 0, 1, 2]), batch_size=3, shuffle=True)
    validation_loader = DataLoader(TinyClassificationDataset([0, 1, 2]), batch_size=3)
    return train_loader, validation_loader


def _run_tiny_study(tmp_path, **kwargs):
    arguments = {
        "trial_configs": _study_configs(),
        "model_factory": lambda config: TinyClassifier(),
        "loader_factory": _study_loaders,
        "criterion_factory": lambda config: nn.CrossEntropyLoss(),
        "device": "cpu", "num_classes": 3,
        "model_dir": tmp_path / "models", "report_dir": tmp_path / "reports",
        "max_epochs": 1, "patience": 1, "warmup_ratio": 0.0,
        "context": {"input_hashes": {"train": "fixture-hash"}},
    }
    arguments.update(kwargs)
    return run_parameter_study(**arguments)


def test_parameter_study_ranks_macro_f1_before_accuracy_loss_and_id():
    rows = pd.DataFrame([
        {"trial_id": "z", "validation_macro_f1": 0.8, "validation_accuracy": 0.7, "validation_loss": 0.4},
        {"trial_id": "a", "validation_macro_f1": 0.8, "validation_accuracy": 0.7, "validation_loss": 0.4},
        {"trial_id": "loss", "validation_macro_f1": 0.8, "validation_accuracy": 0.7, "validation_loss": 0.5},
        {"trial_id": "accuracy", "validation_macro_f1": 0.8, "validation_accuracy": 0.6, "validation_loss": 0.1},
        {"trial_id": "macro", "validation_macro_f1": 0.7, "validation_accuracy": 1.0, "validation_loss": 0.0},
    ])
    ranked = rank_parameter_trials(rows)

    assert ranked["trial_id"].tolist() == ["a", "z", "loss", "accuracy", "macro"]
    assert ranked["rank"].tolist() == [1, 2, 3, 4, 5]
    assert ranked["is_selected"].tolist() == [True, False, False, False, False]
    assert "rank" not in rows


@pytest.mark.parametrize("metric,value", [("validation_macro_f1", float("nan")), ("validation_accuracy", 1.1), ("validation_loss", -0.1)])
def test_parameter_ranking_rejects_invalid_metrics(metric, value):
    row = {"trial_id": "a", "validation_macro_f1": 0.8, "validation_accuracy": 0.7, "validation_loss": 0.4}
    row[metric] = value
    with pytest.raises(ValueError):
        rank_parameter_trials(pd.DataFrame([row]))


def test_parameter_study_keeps_only_winner_and_reloads_matching_metrics(tmp_path):
    results = _run_tiny_study(tmp_path)
    model_dir, report_dir = tmp_path / "models", tmp_path / "reports"
    selected = results.iloc[0]
    assert results["is_selected"].sum() == 1
    assert len(results) == 2
    assert list(model_dir.rglob("*.pt")) == [model_dir / "best_model.pt"]
    assert list(tmp_path.rglob("*.part")) == []

    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model = TinyClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    _, validation_loader = _study_loaders({})
    metrics = evaluate_classifier(model, validation_loader, "cpu", 3)
    for metric in ("loss", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
        assert selected[f"validation_{metric}"] == pytest.approx(metrics[metric])
    metadata = json.loads((model_dir / "best_trial.json").read_text(encoding="utf-8"))
    assert metadata["trial_id"] == selected["trial_id"] == results.attrs["best_trial_id"]
    assert metadata["config"] == results.attrs["best_config"]
    assert metadata["best_epoch"] == checkpoint["epoch"]
    assert metadata["checkpoint_filename"] == "best_model.pt"
    assert metadata["study_context"]["context"]["input_hashes"]["train"] == "fixture-hash"
    saved_results = pd.read_csv(report_dir / "study_results.csv")
    pd.testing.assert_frame_equal(saved_results, results, check_dtype=False)
    assert json.loads((report_dir / "study_context.json").read_text())["seed"] == 42
    for trial in _study_configs():
        trial_dir = report_dir / "trials" / trial["trial_id"]
        assert {path.name for path in trial_dir.iterdir()} == {
            "trial_config.json", "training_history.csv", "validation_metrics.json"
        }
        assert len(pd.read_csv(trial_dir / "training_history.csv")) == 1


def test_parameter_study_reseeds_and_releases_model_optimizer_and_loaders(tmp_path, monkeypatch):
    model_refs, optimizer_refs, loader_refs = [], [], []
    initial_weights, shuffled_inputs = [], []
    real_optimizer = training_helpers.make_optimizer

    def model_factory(config):
        assert all(reference() is None for reference in model_refs + optimizer_refs + loader_refs)
        model = TinyClassifier()
        initial_weights.append(model.backbone.weight.detach().clone())
        model_refs.append(weakref.ref(model))
        return model

    def loader_factory(config):
        loaders = _study_loaders(config)
        loader_refs.extend(weakref.ref(loader) for loader in loaders)
        return loaders

    def record_optimizer(*args, **kwargs):
        optimizer = real_optimizer(*args, **kwargs)
        assert len(optimizer.state) == 0
        optimizer_refs.append(weakref.ref(optimizer))
        return optimizer

    real_fit = training_helpers.fit_classifier

    def record_fit(model, train_loader, *args, **kwargs):
        # Iterating consumes RNG identically in each trial and checks loader shuffle isolation.
        shuffled_inputs.append(next(iter(train_loader))["input_ids"].clone())
        return real_fit(model, train_loader, *args, **kwargs)

    monkeypatch.setattr(training_helpers, "make_optimizer", record_optimizer)
    monkeypatch.setattr(training_helpers, "fit_classifier", record_fit)
    _run_tiny_study(tmp_path, model_factory=model_factory, loader_factory=loader_factory)

    torch.testing.assert_close(initial_weights[0], initial_weights[1])
    torch.testing.assert_close(shuffled_inputs[0], shuffled_inputs[1])
    assert all(reference() is None for reference in model_refs + optimizer_refs + loader_refs)


@pytest.mark.parametrize("case", ["unsafe_id", "reserved_id", "duplicate_id", "case_collision", "nan", "negative_lr", "dropout", "extra_key", "too_many"])
def test_parameter_study_validates_all_configs_before_creating_directories(tmp_path, case):
    configs = _study_configs()
    if case == "unsafe_id":
        configs[1]["trial_id"] = "../escape"
    elif case == "reserved_id":
        configs[1]["trial_id"] = "CON"
    elif case == "duplicate_id":
        configs[1]["trial_id"] = configs[0]["trial_id"]
    elif case == "case_collision":
        configs[1]["trial_id"] = configs[0]["trial_id"].upper()
    elif case == "nan":
        configs[1]["weight_decay"] = float("nan")
    elif case == "negative_lr":
        configs[1]["head_lr"] = -1
    elif case == "dropout":
        configs[1]["dropout"] = 1
    elif case == "extra_key":
        configs[1]["batch_size"] = 8
    else:
        configs = [dict(configs[0], trial_id=f"trial_{index}") for index in range(7)]

    with pytest.raises(ValueError):
        _run_tiny_study(tmp_path, trial_configs=configs)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing_directory", ["models", "reports"])
def test_parameter_study_refuses_to_overwrite_existing_outputs(tmp_path, existing_directory):
    output_dir = tmp_path / existing_directory
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("previous run", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _run_tiny_study(tmp_path)
    assert marker.read_text(encoding="utf-8") == "previous run"
    assert list(tmp_path.iterdir()) == [output_dir]


def test_parameter_study_failure_preserves_completed_results_and_best_checkpoint(tmp_path):
    def model_factory(config):
        if config["trial_id"] == "lower_dropout":
            raise RuntimeError("simulated second-trial failure")
        return TinyClassifier()

    with pytest.raises(RuntimeError, match="second-trial failure"):
        _run_tiny_study(tmp_path, model_factory=model_factory)

    rows = pd.read_csv(tmp_path / "reports/study_results.csv")
    assert rows["trial_id"].tolist() == ["baseline"]
    assert rows["is_selected"].tolist() == [True]
    metadata = json.loads((tmp_path / "models/best_trial.json").read_text())
    assert metadata["trial_id"] == "baseline"
    assert (tmp_path / "models/best_model.pt").is_file()
    assert len(list((tmp_path / "models").rglob("*.pt"))) == 1
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("setting,value", [("seed", -1), ("max_epochs", 0), ("patience", 0), ("num_classes", 0), ("warmup_ratio", 1), ("min_delta", float("inf"))])
def test_parameter_study_rejects_invalid_settings_before_writes(tmp_path, setting, value):
    with pytest.raises(ValueError):
        _run_tiny_study(tmp_path, **{setting: value})
    assert list(tmp_path.iterdir()) == []


def test_parameter_study_rejects_unserializable_context_before_writes(tmp_path):
    with pytest.raises((TypeError, ValueError)):
        _run_tiny_study(tmp_path, context={"not_serializable": object()})
    assert list(tmp_path.iterdir()) == []


def test_parameter_study_has_no_test_loader_or_test_selection_arguments():
    import inspect

    parameter_names = inspect.signature(run_parameter_study).parameters
    assert not any("test" in name for name in parameter_names)


def test_notebook_declares_six_fixed_study_configurations():
    namespace = {
        "pd": pd, "display": lambda value: None, "RUN_PARAMETER_STUDY": True,
    }
    exec(_notebook_cell_source("news05-study-configs"), namespace)
    configs = namespace["trial_configs"]
    expected = [
        ("baseline", 2e-5, 1e-3, 0.3, 0.01),
        ("head_lr", 2e-5, 3e-4, 0.3, 0.01),
        ("backbone_lr", 1e-5, 1e-3, 0.3, 0.01),
        ("both_lr", 1e-5, 3e-4, 0.3, 0.01),
        ("dropout", 2e-5, 1e-3, 0.5, 0.01),
        ("weight_decay", 2e-5, 1e-3, 0.3, 0.05),
    ]
    assert [tuple(config[column] for column in namespace["trial_columns"])
            for config in configs] == expected
    assert len({config["trial_id"] for config in configs}) == 6


def test_notebook_disabled_study_does_not_display_unused_configurations():
    displayed_tables = []
    namespace = {
        "pd": pd, "display": displayed_tables.append, "RUN_PARAMETER_STUDY": False,
    }

    exec(_notebook_cell_source("news05-study-configs"), namespace)

    assert displayed_tables == []
    assert len(namespace["trial_configs"]) == 6


def test_notebook_study_loaders_use_independent_reproducible_generators():
    dataset = TinyClassificationDataset([0, 1, 2] * 4)
    namespace = {
        "DataLoader": DataLoader, "torch": torch, "train_dataset": dataset,
        "validation_dataset": dataset, "BATCH_SIZE": 4, "RANDOM_SEED": 42,
        "DEVICE": torch.device("cpu"),
    }
    exec(_notebook_cell_source("news05-study-loader-factory"), namespace)
    first_train, first_validation = namespace["build_trial_loaders"]({})
    second_train, second_validation = namespace["build_trial_loaders"]({})
    assert first_train is not second_train
    assert first_validation is not second_validation
    assert first_train.generator is not second_train.generator
    assert first_train.num_workers == first_validation.num_workers == 0
    assert isinstance(first_train.sampler, torch.utils.data.RandomSampler)
    initial_state = second_train.generator.get_state().clone()
    first_order = torch.cat([batch["input_ids"][:, 0] for batch in first_train])
    torch.testing.assert_close(second_train.generator.get_state(), initial_state)
    second_order = torch.cat([batch["input_ids"][:, 0] for batch in second_train])
    torch.testing.assert_close(first_order, second_order)
    validation_order = torch.cat([batch["input_ids"][:, 0] for batch in first_validation])
    torch.testing.assert_close(validation_order, torch.arange(3, 15))


def test_notebook_study_release_disabled_never_accesses_runtime_objects():
    namespace = {"RUN_PARAMETER_STUDY": False}
    exec(_notebook_cell_source("news05-study-release"), namespace)
    assert "model" not in namespace


def test_notebook_disabled_study_run_resets_results_without_starting_trials():
    def unexpected_study(*args, **kwargs):
        pytest.fail("Disabled study must not start parameter trials.")

    namespace = {
        "RUN_PARAMETER_STUDY": False, "study_results": object(),
        "run_parameter_study": unexpected_study,
    }

    exec(_notebook_cell_source("news05-study-run"), namespace)

    assert namespace["study_results"] is None


@pytest.mark.parametrize("cell_id", [
    "news05-study-table", "news05-study-comparison",
    "news05-selected-provenance", "news05-selected-model", "news05-selected-evaluation",
    "news05-selected-curves", "news05-selected-save",
])
def test_notebook_selected_study_cells_skip_undefined_results_when_disabled(cell_id):
    namespace = {"RUN_PARAMETER_STUDY": False}

    exec(_notebook_cell_source(cell_id), namespace)

    assert "study_results" not in namespace
    assert "model" not in namespace
    assert "RUN_MODEL_DIR" not in namespace


def test_notebook_selected_state_discards_stale_study_without_touching_current_run():
    def unexpected_access(*args, **kwargs):
        pytest.fail("Disabled study must not read or restore an old checkpoint.")

    model, validation_results, validation_metrics = object(), object(), object()
    model_dir, report_dir = Path("current-model"), Path("current-report")
    namespace = {
        "RUN_PARAMETER_STUDY": False, "study_results": object(),
        "model": model, "validation_results": validation_results,
        "validation_metrics": validation_metrics,
        "RUN_MODEL_DIR": model_dir, "RUN_REPORT_DIR": report_dir,
        "build_trial_model": unexpected_access,
        "torch": SimpleNamespace(load=unexpected_access),
    }

    for cell_id in (
        "news05-selected-state", "news05-selected-provenance", "news05-selected-model",
        "news05-selected-evaluation", "news05-selected-curves", "news05-selected-save",
    ):
        exec(_notebook_cell_source(cell_id), namespace)

    assert namespace["study_results"] is None
    assert namespace["model"] is model
    assert namespace["validation_results"] is validation_results
    assert namespace["validation_metrics"] is validation_metrics
    assert namespace["RUN_MODEL_DIR"] is model_dir
    assert namespace["RUN_REPORT_DIR"] is report_dir


@pytest.mark.parametrize("include_results", [False, True])
def test_notebook_selected_state_requires_completed_enabled_study(include_results):
    namespace = {"RUN_PARAMETER_STUDY": True}
    if include_results:
        namespace["study_results"] = None

    with pytest.raises(RuntimeError, match="RUN_PARAMETER_STUDY"):
        exec(_notebook_cell_source("news05-selected-state"), namespace)


@pytest.fixture
def selected_study_namespace(monkeypatch, tmp_path):
    project_dir = Path(__file__).resolve().parents[1]
    model_dir = tmp_path / "models" / "studies" / "current-study"
    report_dir = tmp_path / "reports" / "studies" / "current-study"
    run_config = {
        "input_sha256": {"train": "current-train", "validation": "current-validation"},
        "label_mapping": {"label2id": {"industry": 0, "mention_only": 1}},
        "max_length": 256, "head_ratio": 0.75,
    }
    saved_context = json.loads(json.dumps(run_config))
    saved_context["training_code_sha256"] = hashlib.sha256(
        (project_dir / "src/news_type_training.py").read_bytes()
    ).hexdigest()
    trial_config = {"trial_id": "dropout", "dropout": 0.5}
    metadata = {
        "study_context": {"context": saved_context},
        "config": dict(trial_config), "trial_id": "dropout",
    }
    study_results = pd.DataFrame([{"trial_id": "dropout"}])
    study_results.attrs.update(
        checkpoint_path=str(model_dir / "best_model.pt"),
        best_config=dict(trial_config), best_trial_id="dropout",
    )
    original_read_text = Path.read_text

    def read_metadata(path, *args, **kwargs):
        if path == (model_dir / "best_trial.json").resolve():
            return json.dumps(metadata)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_metadata)
    namespace = {
        "RUN_PARAMETER_STUDY": True, "study_results": study_results,
        "Path": Path, "json": json, "hashlib": hashlib, "PROJECT_DIR": project_dir,
        "run_config": run_config, "fixed_parameter_names": tuple(run_config),
        "input_file_hashes": dict(run_config["input_sha256"]),
        "STUDY_MODEL_DIR": model_dir, "STUDY_REPORT_DIR": report_dir,
    }
    return namespace, metadata


@pytest.mark.parametrize("mismatch", [
    "checkpoint_path", "saved_data", "current_data", "label_mapping", "training_code",
    "trial_config", "trial_id",
])
def test_notebook_selected_provenance_rejects_mismatch_before_model_restore(
    selected_study_namespace, mismatch,
):
    namespace, metadata = selected_study_namespace
    if mismatch == "checkpoint_path":
        namespace["study_results"].attrs["checkpoint_path"] = str(Path("old/best_model.pt"))
    elif mismatch == "saved_data":
        metadata["study_context"]["context"]["input_sha256"]["train"] = "old-train"
    elif mismatch == "current_data":
        namespace["input_file_hashes"]["train"] = "different-current-train"
    elif mismatch == "label_mapping":
        metadata["study_context"]["context"]["label_mapping"]["label2id"]["industry"] = 1
    elif mismatch == "training_code":
        metadata["study_context"]["context"]["training_code_sha256"] = "old-code"
    elif mismatch == "trial_config":
        metadata["config"]["dropout"] = 0.3
    elif mismatch == "trial_id":
        metadata["trial_id"] = "old-trial"
    model = object()
    namespace["model"] = model

    def unexpected_restore(*args, **kwargs):
        pytest.fail("Invalid provenance must be rejected before loading model weights.")

    namespace["build_trial_model"] = unexpected_restore
    namespace["torch"] = SimpleNamespace(load=unexpected_restore)

    with pytest.raises(RuntimeError):
        exec(_notebook_cell_source("news05-selected-provenance"), namespace)
        exec(_notebook_cell_source("news05-selected-model"), namespace)

    assert namespace["model"] is model


def test_notebook_selected_provenance_allows_matching_study_context(selected_study_namespace):
    namespace, metadata = selected_study_namespace

    exec(_notebook_cell_source("news05-selected-state"), namespace)
    exec(_notebook_cell_source("news05-selected-provenance"), namespace)

    assert namespace["selected_checkpoint"] == (
        namespace["STUDY_MODEL_DIR"] / "best_model.pt"
    ).resolve()
    assert namespace["saved_study_context"] == metadata["study_context"]["context"]
    assert "model" not in namespace


@pytest.mark.parametrize("use_revision", [False, True])
def test_notebook_cold_review_restores_once_and_exports_validation_without_training(
    tmp_path, notebook_plotting, validation_review_inputs, use_revision,
):
    from datetime import datetime
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score

    namespace = _validation_review_notebook_namespace(validation_review_inputs)
    validation_data, train_data, _, label2id, token_statistics = validation_review_inputs
    for key in ("validation_results", "validation_dataset", "validation_loader", "validation_encodings"):
        namespace.pop(key)
    events = []

    def unexpected_training(*args, **kwargs):
        pytest.fail("Review mode must not train, build an optimizer, or run parameter trials.")

    namespace.update({
        "Path": Path, "torch": torch, "nn": nn, "json": json, "hashlib": hashlib,
        "datetime": datetime, "plt": notebook_plotting, "display": lambda value: None,
        "ConfusionMatrixDisplay": ConfusionMatrixDisplay,
        "classification_report": classification_report, "confusion_matrix": confusion_matrix,
        "MODEL_DIR": tmp_path / "models", "REPORT_DIR": tmp_path / "reports",
        "MODEL_CACHE_DIR": tmp_path / "cache", "DEVICE": torch.device("cpu"),
        "make_optimizer": unexpected_training, "make_scheduler": unexpected_training,
        "fit_classifier": unexpected_training, "run_parameter_study": unexpected_training,
        "evaluate_classifier": evaluate_classifier, "DataLoader": DataLoader,
        "EncodedNewsDataset": EncodedNewsDataset, "PhoBertCNN": PhoBertCNN,
    })
    for cell_id in ("news05-07", "news05-08", "news05-run-mode"):
        exec(_notebook_cell_source(cell_id), namespace)
    assert namespace["RUN_MODE"] == "review"
    assert namespace["RUN_PARAMETER_STUDY"] is False
    assert namespace["EVALUATE_TEST"] is False
    model_dir, report_dir = namespace["SAVED_MODEL_DIR"], namespace["SAVED_REPORT_DIR"]
    for directory in (model_dir / "tokenizer", model_dir / "backbone", report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    protected_paths = {
        model_dir / "best_model.pt": b"existing checkpoint",
        model_dir / "run_config.json": b"existing configuration",
        report_dir / "training_history.csv": b"existing history",
        report_dir / "validation_metrics.json": b"existing metrics",
    }
    for path, content in protected_paths.items():
        path.write_bytes(content)
    namespace.update(NUM_CLASSES=len(label2id), NUM_FILTERS=2, GLOBAL_PROJECTION_SIZE=3)
    namespace["ordered_labels"] = list(label2id)
    namespace["label_mapping"] = {"label2id": label2id, "label_policy_version": "company_awards_listing_v2"}
    namespace["class_weights_table"] = pd.DataFrame({
        "label_id": list(label2id.values()), "class_weight": [1.0, 1.2, 1.5, 2.0],
    })
    namespace["input_file_hashes"] = {"fixture": "current data"}
    best_f1 = f1_score(
        validation_data["label_id"], [0] * len(validation_data),
        labels=list(label2id.values()), average="macro", zero_division=0,
    )
    original_best_f1 = 0.8 if use_revision else best_f1
    history = pd.DataFrame({
        "epoch": [1, 2], "train_loss": [1.2, 0.8], "validation_loss": [1.3, 1.0],
        "train_accuracy": [0.3, 0.5], "validation_accuracy": [0.25, 0.375],
        "train_macro_f1": [0.2, 0.4], "validation_macro_f1": [0.1, original_best_f1],
    })
    history.attrs.update(best_epoch=2, best_validation_macro_f1=original_best_f1)
    saved_config = {
        "best_epoch": 2, "best_validation_macro_f1": original_best_f1,
        "input_sha256": {"fixture": "baseline data"},
        "label_mapping": {**namespace["label_mapping"], "label_policy_version": (
            "gold_price_industry_v1" if use_revision else "company_awards_listing_v2"
        )},
    }
    revision_path = tmp_path / "validation_revision.json"
    namespace.update(VALIDATION_REVISION_PATH=revision_path, SPLIT_DIR=tmp_path / "splits")
    if use_revision:
        manifest = {
            "revision_id": training_helpers.VALIDATION_REVISION_ID,
            "pending_reviews": [{"news_id": news_id, "reason": "Pending human decision."}
                                for news_id in sorted(training_helpers._REVIEW_PENDING_IDS)],
        }
        revision_path.write_text(json.dumps(manifest), encoding="utf-8")
        exec(_notebook_cell_source("news05-validation-revision"), namespace)
        history.attrs["evaluation_revision"] = training_helpers.VALIDATION_REVISION_ID
    else:
        namespace["validation_revision"] = None

    def load_metadata(received_model_dir, received_report_dir, **kwargs):
        events.append("metadata")
        assert received_model_dir == model_dir and received_report_dir == report_dir
        assert kwargs["input_sha256"] == namespace["input_file_hashes"]
        assert kwargs["label_mapping"] == namespace["label_mapping"]
        assert kwargs["validation_revision"] == namespace["validation_revision"]
        assert kwargs["validation_path"] == namespace["SPLIT_DIR"] / "validation.csv"
        assert kwargs["expected_config"]["dropout"] == 0.5
        assert kwargs["expected_config"]["class_weights_by_id"] == torch.tensor(
            [1.0, 1.2, 1.5, 2.0], dtype=torch.float32,
        ).tolist()
        return saved_config, history

    namespace["load_saved_run_metadata"] = load_metadata
    exec(_notebook_cell_source("news05-saved-metadata"), namespace)
    resource_dir = tmp_path / "segmenter"
    resource_dir.mkdir()
    resource_names = ("jar", "vocabulary", "rules")
    for name in resource_names:
        (resource_dir / name).write_bytes(name.encode("ascii"))
    saved_config["segmenter_sha256"] = {
        name: hashlib.sha256((resource_dir / name).read_bytes()).hexdigest() for name in resource_names
    }
    namespace.update(
        VNCORENLP_DIR=resource_dir, vncorenlp_resources=resource_names,
        validate_saved_segmenter=training_helpers.validate_saved_segmenter,
    )
    exec(_notebook_cell_source("news05-segmenter-check"), namespace)

    def load_tokenizer(path, **kwargs):
        events.append("tokenizer")
        assert Path(path) == model_dir / "tokenizer" and kwargs["local_files_only"] is True
        return FakeTokenizer()

    def load_backbone_config(path, **kwargs):
        events.append("backbone-config")
        assert Path(path) == model_dir / "backbone" and kwargs["local_files_only"] is True
        return SimpleNamespace(hidden_size=8)

    namespace["AutoTokenizer"] = SimpleNamespace(from_pretrained=load_tokenizer)
    namespace["AutoConfig"] = SimpleNamespace(from_pretrained=load_backbone_config)
    namespace["AutoModel"] = SimpleNamespace(
        from_config=lambda config: TinyBackbone(config.hidden_size),
        from_pretrained=unexpected_training,
    )
    exec(_notebook_cell_source("news05-20"), namespace)
    basic_encoding = {"input_ids": [0, 10, 11, 12, 13, 14, 2, 1], "attention_mask": [1] * 7 + [0]}
    namespace["train_encodings"] = [dict(basic_encoding) for _ in range(len(train_data))]
    namespace["validation_encodings"] = [
        {**basic_encoding, **statistics} for statistics in token_statistics
    ]
    exec(_notebook_cell_source("news05-25"), namespace)
    exec(_notebook_cell_source("news05-27"), namespace)

    def restore_model(model, received_model_dir, config, device):
        events.append("restore")
        assert received_model_dir == model_dir and config is saved_config
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()
            model.classifier.bias[0] = 1.0
        model.to(device).eval()
        return {"epoch": 2, "validation_macro_f1": original_best_f1}

    namespace["restore_saved_checkpoint"] = restore_model
    for cell_id in (
        "news05-restore-saved", "news05-28", "news05-30", "news05-31", "news05-33",
        "news05-34", "news05-35", "news05-36", "news05-baseline-curves", "news05-38",
        "news05-review-score-check", "news05-39", "news05-41", "news05-42", "news05-43",
        "news05-revision-config", "news05-revision-results",
        "news05-selected-state", "news05-selected-provenance", "news05-selected-model",
        "news05-selected-evaluation", "news05-selected-curves", "news05-selected-metrics",
        "news05-selected-confusion", "news05-selected-class-f1", "news05-selected-save",
        "news05-review-order", "news05-review-build", "news05-review-summary",
        "news05-review-pairs", "news05-review-articles", "news05-review-save",
        "news05-45", "news05-46", "news05-47", "news05-48",
        "news05-final-metrics", "news05-final-class-f1",
    ):
        exec(_notebook_cell_source(cell_id), namespace)

    assert events == ["metadata", "tokenizer", "backbone-config", "restore"]
    assert "optimizer" not in namespace and "scheduler" not in namespace
    assert "test_data" not in namespace and "test_results" not in namespace
    assert namespace["run_config"] is saved_config
    assert namespace["history"] is history
    assert namespace["validation_metrics"]["macro_f1"] == pytest.approx(best_f1)
    assert list(namespace["validation_results"]["predictions"]) == [0] * len(validation_data)
    for path, original_content in protected_paths.items():
        assert path.read_bytes() == original_content
    result_dir = (namespace["REPORT_DIR"] / "evaluations" / training_helpers.VALIDATION_REVISION_ID
                  if use_revision else report_dir)
    assert namespace["RUN_REPORT_DIR"] == result_dir
    assert namespace["SAVED_REPORT_DIR"] == report_dir
    assert namespace["RUN_MODEL_DIR"] == model_dir
    assert {path.name for path in result_dir.glob("*.png")} == {
        "selected_validation_scores.png", "validation_confusion_matrix.png", "validation_class_f1.png",
    }
    assert (result_dir / "validation_review_summary.csv").is_file()
    assert (result_dir / "validation_errors.csv").is_file()
    if use_revision:
        assert {path.name for path in result_dir.iterdir()} == {
            "evaluation_config.json", "validation_metrics.json", "validation_classification_report.csv",
            "selected_validation_scores.png", "validation_confusion_matrix.png", "validation_class_f1.png",
            "validation_review_summary.csv", "validation_errors.csv",
        }
        evaluation = json.loads((result_dir / "evaluation_config.json").read_text(encoding="utf-8"))
        assert evaluation["model_retrained"] is False and evaluation["test_evaluated"] is False
        assert evaluation["final_evaluation"] is False and len(evaluation["pending_reviews"]) == 2
        assert evaluation["training_label_policy"] == "gold_price_industry_v1"
        assert evaluation["evaluation_label_policy"] == "company_awards_listing_v2"
        assert evaluation["checkpoint_epoch"] == 2
        assert json.loads((result_dir / "validation_metrics.json").read_text(encoding="utf-8"))["macro_f1"] == pytest.approx(best_f1)
        assert original_best_f1 != best_f1


def test_notebook_train_mode_still_builds_optimizer_and_calls_fit(tmp_path):
    events = []
    expected_history = object()

    def make_optimizer_spy(model, **kwargs):
        events.append(("optimizer", kwargs))
        return "optimizer fixture"

    def make_scheduler_spy(optimizer, **kwargs):
        events.append(("scheduler", kwargs))
        assert optimizer == "optimizer fixture"
        return "scheduler fixture"

    def fit_spy(**kwargs):
        events.append(("fit", kwargs))
        return expected_history

    namespace = {
        "RUN_MODE": "train", "make_optimizer": make_optimizer_spy,
        "make_scheduler": make_scheduler_spy, "fit_classifier": fit_spy,
        "model": object(), "train_loader": [1, 2, 3], "validation_loader": object(),
        "criterion": object(), "DEVICE": torch.device("cpu"), "NUM_CLASSES": 4,
        "BACKBONE_LR": 2e-5, "HEAD_LR": 1e-3, "WEIGHT_DECAY": 0.01,
        "MAX_EPOCHS": 8, "WARMUP_RATIO": 0.1, "PATIENCE": 2, "MIN_DELTA": 1e-4,
        "CHECKPOINT_PATH": tmp_path / "best_model.pt",
    }

    for cell_id in ("news05-31", "news05-34"):
        exec(_notebook_cell_source(cell_id), namespace)

    assert [name for name, _ in events] == ["optimizer", "scheduler", "fit"]
    assert events[1][1]["total_steps"] == 24
    assert events[2][1]["checkpoint_path"] == namespace["CHECKPOINT_PATH"]
    assert events[2][1]["optimizer"] == "optimizer fixture"
    assert events[2][1]["scheduler"] == "scheduler fixture"
    assert namespace["history"] is expected_history


def test_notebook_review_rejects_changed_validation_score_before_report_exports():
    namespace = {
        "RUN_MODE": "review", "run_config": {"best_validation_macro_f1": 0.88},
        "validation_metrics": {"macro_f1": 0.75}, "history": pd.DataFrame(),
    }

    with pytest.raises(RuntimeError, match="Macro-F1"):
        exec(_notebook_cell_source("news05-review-score-check"), namespace)


@pytest.mark.parametrize("pending_count", [0, 2])
def test_notebook_approved_revision_allows_recomputed_score_without_changing_saved_selection(pending_count):
    original_config = {"best_epoch": 2, "best_validation_macro_f1": 0.88}
    history = pd.DataFrame()
    history.attrs["evaluation_revision"] = training_helpers.VALIDATION_REVISION_ID
    namespace = {
        "RUN_MODE": "review", "run_config": original_config,
        "validation_metrics": {"macro_f1": 0.75}, "history": history,
        "validation_revision": {"pending_reviews": [
            {"news_id": news_id, "reason": "Awaiting a human decision."}
            for news_id in sorted(training_helpers._REVIEW_PENDING_IDS)[:pending_count]
        ]},
    }

    exec(_notebook_cell_source("news05-review-score-check"), namespace)

    assert original_config == {"best_epoch": 2, "best_validation_macro_f1": 0.88}
    assert namespace["validation_metrics"]["macro_f1"] == 0.75


@pytest.mark.parametrize("mode", ["train", "review"])
def test_notebook_pending_revision_blocks_test_evaluation_in_every_mode(tmp_path, mode):
    path = tmp_path / "revision.json"
    path.write_text(json.dumps({
        "revision_id": training_helpers.VALIDATION_REVISION_ID,
        "pending_reviews": [{"news_id": news_id, "reason": "Still unresolved."}
                            for news_id in sorted(training_helpers._REVIEW_PENDING_IDS)],
    }), encoding="utf-8")
    namespace = {"RUN_MODE": mode, "EVALUATE_TEST": True, "VALIDATION_REVISION_PATH": path, "json": json}

    with pytest.raises(RuntimeError, match="test"):
        exec(_notebook_cell_source("news05-validation-revision"), namespace)

    assert "test_results" not in namespace


@pytest.mark.parametrize("mode", ["train", "review"])
@pytest.mark.parametrize("evaluate_test", [False, True])
def test_notebook_resolved_reviews_allow_only_explicit_test_opt_in(tmp_path, mode, evaluate_test):
    settings = {}
    exec(_notebook_cell_source("news05-07"), settings)
    assert settings["EVALUATE_TEST"] is False
    revision = {
        "revision_id": training_helpers.VALIDATION_REVISION_ID,
        "pending_reviews": [],
        "resolved_reviews": [
            {
                "news_id": news_id, "news_type": "financial_results",
                "decision": "confirmed_unchanged", "reviewed_date": "2026-10-25",
                "reason": "Human review confirms the primary topic is financial results.",
            }
            for news_id in sorted(training_helpers._REVIEW_PENDING_IDS)
        ],
    }
    path = tmp_path / "revision.json"
    path.write_text(json.dumps(revision), encoding="utf-8")
    namespace = {
        "RUN_MODE": mode, "EVALUATE_TEST": evaluate_test,
        "VALIDATION_REVISION_PATH": path, "json": json,
    }

    exec(_notebook_cell_source("news05-validation-revision"), namespace)

    assert namespace["validation_revision"] == revision
    assert namespace["EVALUATE_TEST"] is evaluate_test
    assert "test_results" not in namespace and "test_loader" not in namespace


@pytest.mark.parametrize("mode, required", [("review", True), ("train", False)])
def test_notebook_revision_manifest_is_required_for_review_but_optional_for_train(tmp_path, mode, required):
    namespace = {
        "RUN_MODE": mode, "EVALUATE_TEST": False, "json": json,
        "VALIDATION_REVISION_PATH": tmp_path / "missing.json",
    }
    if required:
        with pytest.raises(FileNotFoundError):
            exec(_notebook_cell_source("news05-validation-revision"), namespace)
    else:
        exec(_notebook_cell_source("news05-validation-revision"), namespace)
        assert namespace["validation_revision"] is None


@pytest.mark.parametrize("use_revision", [False, True])
def test_notebook_revision_report_routing_preserves_original_run_paths(tmp_path, use_revision):
    original_model = tmp_path / "models/runs/original"
    original_report = tmp_path / "reports/runs/original"
    original_report.mkdir(parents=True)
    protected = original_report / "validation_metrics.json"
    protected.write_bytes(b"original metric bytes")
    history = pd.DataFrame()
    if use_revision:
        history.attrs["evaluation_revision"] = training_helpers.VALIDATION_REVISION_ID
    namespace = {
        "RUN_MODE": "review", "SAVED_MODEL_DIR": original_model, "SAVED_REPORT_DIR": original_report,
        "MODEL_DIR": tmp_path / "models", "REPORT_DIR": tmp_path / "reports",
        "RUN_NAME": "original", "history": history,
    }

    exec(_notebook_cell_source("news05-33"), namespace)

    expected = (tmp_path / "reports/evaluations" / training_helpers.VALIDATION_REVISION_ID
                if use_revision else original_report)
    assert namespace["RUN_REPORT_DIR"] == expected and expected.is_dir()
    assert namespace["RUN_MODEL_DIR"] == namespace["SAVED_MODEL_DIR"] == original_model
    assert namespace["SAVED_REPORT_DIR"] == original_report
    assert protected.read_bytes() == b"original metric bytes"


def test_notebook_study_release_rejects_cpu_before_moving_model():
    namespace = {"RUN_PARAMETER_STUDY": True, "DEVICE": torch.device("cpu")}
    with pytest.raises(RuntimeError, match="GPU"):
        exec(_notebook_cell_source("news05-study-release"), namespace)


def test_notebook_study_release_moves_baseline_and_drops_optimizer_on_gpu():
    events = []
    cpu_logits = object()
    namespace = {
        "RUN_PARAMETER_STUDY": True, "DEVICE": SimpleNamespace(type="cuda"),
        "model": SimpleNamespace(to=lambda device: events.append(("model", device))),
        "example_logits": SimpleNamespace(cpu=lambda: cpu_logits),
        "optimizer": object(), "scheduler": object(),
        "gc": SimpleNamespace(collect=lambda: events.append("collect")),
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: events.append("empty"))),
    }
    exec(_notebook_cell_source("news05-study-release"), namespace)
    assert events == [("model", "cpu"), "collect", "empty"]
    assert namespace["example_logits"] is cpu_logits
    assert "optimizer" not in namespace and "scheduler" not in namespace


@pytest.fixture
def notebook_plotting(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")
    yield plt
    plt.close("all")


def test_notebook_training_curves_show_loss_accuracy_f1_and_selected_epoch(notebook_plotting):
    history = pd.DataFrame({
        "epoch": [1, 2, 3], "train_loss": [1.4, 0.7, 0.3],
        "validation_loss": [1.1, 0.8, 0.9], "train_accuracy": [0.5, 0.7, 0.9],
        "validation_accuracy": [0.6, 0.8, 0.7], "train_macro_f1": [0.4, 0.6, 0.8],
        "validation_macro_f1": [0.5, 0.7, 0.6],
    })
    namespace = {"plt": notebook_plotting}
    exec(_notebook_cell_source("news05-36"), namespace)
    figure = namespace["plot_training_history"](history, 2)
    assert len(figure.axes) == 3
    assert [axis.get_title() for axis in figure.axes] == ["Loss", "Accuracy", "Macro-F1"]
    for axis, metric in zip(figure.axes, ("loss", "accuracy", "macro_f1")):
        np.testing.assert_allclose(axis.lines[0].get_ydata(), history["train_" + metric])
        np.testing.assert_allclose(axis.lines[1].get_ydata(), history["validation_" + metric])
        np.testing.assert_allclose(axis.lines[2].get_xdata(), [2, 2])
        assert axis.lines[2].get_linestyle() == "--"


def test_notebook_selected_score_bars_use_evaluated_metrics(tmp_path, notebook_plotting):
    namespace = {
        "pd": pd, "plt": notebook_plotting, "RUN_REPORT_DIR": tmp_path,
        "validation_metrics": {"accuracy": 0.73, "macro_f1": 0.61},
    }
    exec(_notebook_cell_source("news05-selected-metrics"), namespace)
    axis = namespace["axis"]
    assert [bar.get_height() for bar in axis.containers[0]] == pytest.approx([0.73, 0.61])
    assert [label.get_text() for label in axis.get_xticklabels()] == ["Accuracy", "Macro-F1"]
    assert (tmp_path / "selected_validation_scores.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_notebook_final_score_plot_does_not_access_test_when_disabled(notebook_plotting):
    namespace = {"EVALUATE_TEST": False}
    exec(_notebook_cell_source("news05-final-metrics"), namespace)
    assert "final_scores" not in namespace
    assert notebook_plotting.get_fignums() == []


def test_notebook_final_score_plot_separates_validation_and_test(tmp_path, notebook_plotting):
    namespace = {
        "pd": pd, "plt": notebook_plotting, "RUN_REPORT_DIR": tmp_path,
        "EVALUATE_TEST": True,
        "validation_metrics": {"accuracy": 0.81, "macro_f1": 0.74},
        "test_metrics": {"accuracy": 0.77, "macro_f1": 0.68},
    }
    exec(_notebook_cell_source("news05-final-metrics"), namespace)
    axis = namespace["axis"]
    assert [bar.get_height() for bar in axis.containers[0]] == pytest.approx([0.81, 0.74])
    assert [bar.get_height() for bar in axis.containers[1]] == pytest.approx([0.77, 0.68])
    assert [label.get_text() for label in axis.get_xticklabels()] == ["Accuracy", "Macro-F1"]
    assert [label.get_text() for label in axis.get_legend().get_texts()] == ["Validation", "Test"]
    assert (tmp_path / "final_accuracy_f1.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.fixture
def validation_review_inputs():
    label2id = {"alpha": 0, "beta": 1, "gamma": 2, "absent": 3}
    train_labels = [0] * 4 + [1] * 3 + [2] * 2 + [3]
    train_data = pd.DataFrame({
        "news_id": [f"train-{index}" for index in range(len(train_labels))],
        "news_type": [list(label2id)[label] for label in train_labels],
        "label_id": train_labels, "split": "train",
    })
    labels = [0, 0, 0, 1, 1, 2, 2, 2]
    predictions = [1, 2, 0, 0, 1, 0, 0, 2]
    validation_data = pd.DataFrame({
        "news_id": [f"validation-{index}" for index in range(len(labels))],
        "news_type": [list(label2id)[label] for label in labels],
        "label_id": labels, "split": "validation",
        "title": [f"Article {index}" for index in range(len(labels))],
        "content": [f"Body {index}" for index in range(len(labels))],
        "source": [f"Source {index % 3}" for index in range(len(labels))],
        "tickers": [f"T{index}" for index in range(len(labels))],
        "published_date": "2026-01-01",
        "url": [f"https://example.test/news/{index}" for index in range(len(labels))],
    }, index=[80, 10, 70, 20, 60, 30, 50, 40])
    probabilities = np.full((len(labels), len(label2id)), 0.1)
    for index, probability in enumerate([0.61, 0.7, 0.7, 0.8, 0.7, 0.85, 0.65, 0.7]):
        probabilities[index] = (1 - probability) / 3
        probabilities[index, predictions[index]] = probability
    results = {"labels": labels, "predictions": predictions, "probabilities": probabilities}
    encodings = [
        {"original_token_count": count, "retained_token_count": min(count, 254)}
        for count in [12, 300, 15, 500, 25, 700, 1000, 35]
    ]
    return validation_data, train_data, results, label2id, encodings


def test_validation_review_counts_false_negatives_positives_and_class_f1(validation_review_inputs):
    summary, errors, pairs = training_helpers.build_validation_review(*validation_review_inputs)
    classes = summary.set_index("news_type")
    assert summary["news_type"].tolist() == ["absent", "alpha", "gamma", "beta"]
    assert classes.loc["alpha", ["train_count", "validation_count"]].tolist() == [4, 3]
    assert classes.loc["alpha", ["false_negatives", "false_positives"]].tolist() == [2, 3]
    assert classes.loc["alpha", ["precision", "recall", "f1"]].tolist() == pytest.approx([
        1 / 4, 1 / 3, 2 / 7,
    ])
    assert classes.loc["beta", ["false_negatives", "false_positives", "f1"]].tolist() == [1, 1, 0.5]
    assert classes.loc["gamma", ["false_negatives", "false_positives", "f1"]].tolist() == [2, 1, 0.4]
    assert classes.loc["absent", ["train_count", "validation_count", "f1"]].tolist() == [1, 0, 0]
    assert classes.loc["absent", "most_confused_with"] == ""
    assert classes.loc["absent", "confusion_count"] == 0
    assert len(errors) == summary["false_negatives"].sum() == summary["false_positives"].sum() == 5
    assert pairs["article_count"].sum() == 5


def test_validation_review_preserves_directed_confusions_and_breaks_ties_deterministically(
    validation_review_inputs,
):
    summary, _, pairs = training_helpers.build_validation_review(*validation_review_inputs)
    assert list(pairs.itertuples(index=False, name=None)) == [
        ("gamma", "alpha", 2), ("alpha", "beta", 1),
        ("alpha", "gamma", 1), ("beta", "alpha", 1),
    ]
    classes = summary.set_index("news_type")
    assert classes.loc["alpha", ["most_confused_with", "confusion_count"]].tolist() == ["beta", 1]
    assert classes.loc["gamma", ["most_confused_with", "confusion_count"]].tolist() == ["alpha", 2]


def test_validation_review_keeps_article_metadata_probabilities_and_token_counts_aligned(
    validation_review_inputs,
):
    validation_data, _, results, _, encodings = validation_review_inputs
    _, errors, _ = training_helpers.build_validation_review(*validation_review_inputs)
    assert errors["news_id"].tolist() == [
        "validation-5", "validation-3", "validation-1", "validation-6", "validation-0",
    ]
    assert errors.index.tolist() == list(range(5))
    for _, row in errors.iterrows():
        position = int(row["news_id"].rsplit("-", 1)[-1])
        for column in validation_data:
            assert row[column] == validation_data.iloc[position][column]
        probabilities = results["probabilities"][position]
        assert row["predicted_probability"] == pytest.approx(probabilities.max())
        assert row["assigned_label_probability"] == pytest.approx(probabilities[results["labels"][position]])
        assert row["probability_margin"] == pytest.approx(np.sort(probabilities)[-1] - np.sort(probabilities)[-2])
        assert row["original_token_count"] == encodings[position]["original_token_count"]
        assert row["retained_token_count"] == encodings[position]["retained_token_count"]
        assert bool(row["was_truncated"]) == (
            encodings[position]["original_token_count"] > encodings[position]["retained_token_count"]
        )


def test_validation_review_does_not_mutate_frames_predictions_mapping_or_encodings(validation_review_inputs):
    validation_data, train_data, results, label2id, encodings = validation_review_inputs
    saved_validation = validation_data.copy(deep=True)
    saved_train = train_data.copy(deep=True)
    saved_results = {
        "labels": results["labels"].copy(), "predictions": results["predictions"].copy(),
        "probabilities": results["probabilities"].copy(),
    }
    saved_mapping = dict(label2id)
    saved_encodings = [dict(encoding) for encoding in encodings]
    training_helpers.build_validation_review(*validation_review_inputs)
    pd.testing.assert_frame_equal(validation_data, saved_validation)
    pd.testing.assert_frame_equal(train_data, saved_train)
    assert results["labels"] == saved_results["labels"]
    assert results["predictions"] == saved_results["predictions"]
    np.testing.assert_array_equal(results["probabilities"], saved_results["probabilities"])
    assert label2id == saved_mapping
    assert encodings == saved_encodings


def test_validation_review_handles_no_errors_and_optional_token_statistics(validation_review_inputs):
    validation_data, train_data, results, label2id, _ = validation_review_inputs
    results["predictions"] = results["labels"].copy()
    results["probabilities"] = np.eye(len(label2id))[results["labels"]]
    summary, errors, pairs = training_helpers.build_validation_review(
        validation_data, train_data, results, label2id,
    )
    assert errors.empty and pairs.empty
    assert {"news_id", "predicted_news_type", "predicted_probability", "probability_margin"}.issubset(errors)
    assert "was_truncated" not in errors
    assert pairs.columns.tolist() == ["news_type", "predicted_news_type", "article_count"]
    assert summary["false_negatives"].sum() == summary["false_positives"].sum() == 0
    assert summary["most_confused_with"].eq("").all()
    assert summary["confusion_count"].eq(0).all()
    assert summary.set_index("news_type").loc[["alpha", "beta", "gamma"], "f1"].eq(1).all()
    assert summary.set_index("news_type").loc["absent", "f1"] == 0


@pytest.mark.parametrize("case", [
    "validation_is_test", "train_is_validation", "overlap", "duplicate_ids", "empty_ids",
    "unknown_label", "mapping_disagreement", "empty_validation", "missing_required_column",
])
def test_validation_review_rejects_invalid_frames(validation_review_inputs, case):
    validation_data, train_data, results, label2id, _ = validation_review_inputs
    if case == "validation_is_test":
        validation_data["split"] = "test"
    elif case == "train_is_validation":
        train_data["split"] = "validation"
    elif case == "overlap":
        train_data.loc[0, "news_id"] = validation_data.iloc[0]["news_id"]
    elif case == "duplicate_ids":
        validation_data.iloc[1, validation_data.columns.get_loc("news_id")] = validation_data.iloc[0]["news_id"]
    elif case == "empty_ids":
        validation_data.iloc[0, validation_data.columns.get_loc("news_id")] = " "
    elif case == "unknown_label":
        validation_data.iloc[0, validation_data.columns.get_loc("news_type")] = "unknown"
    elif case == "mapping_disagreement":
        validation_data.iloc[0, validation_data.columns.get_loc("label_id")] = 1
    elif case == "empty_validation":
        validation_data = validation_data.iloc[:0]
    else:
        validation_data = validation_data.drop(columns="news_id")
    with pytest.raises(ValueError):
        training_helpers.build_validation_review(validation_data, train_data, results, label2id)


@pytest.mark.parametrize("case", [
    "labels_order", "labels_length", "labels_float", "unknown_prediction", "predictions_bool",
    "probabilities_shape", "probabilities_nan", "probabilities_inf", "probabilities_negative",
    "probabilities_above_one", "probabilities_sum", "prediction_not_argmax",
])
def test_validation_review_rejects_invalid_prediction_alignment_and_probabilities(validation_review_inputs, case):
    validation_data, train_data, results, label2id, _ = validation_review_inputs
    if case == "labels_order":
        results["labels"] = results["labels"][::-1]
    elif case == "labels_length":
        results["labels"] = results["labels"][:-1]
    elif case == "labels_float":
        results["labels"] = np.asarray(results["labels"], dtype=float)
    elif case == "unknown_prediction":
        results["predictions"][0] = len(label2id)
    elif case == "predictions_bool":
        results["predictions"] = np.asarray(results["predictions"], dtype=bool)
    elif case == "probabilities_shape":
        results["probabilities"] = results["probabilities"][:, :-1]
    elif case == "probabilities_nan":
        results["probabilities"][0, 0] = np.nan
    elif case == "probabilities_inf":
        results["probabilities"][0, 0] = np.inf
    elif case == "probabilities_negative":
        results["probabilities"][0, 0] = -0.1
    elif case == "probabilities_above_one":
        results["probabilities"][0, 0] = 1.1
    elif case == "probabilities_sum":
        results["probabilities"][0] *= 0.8
    else:
        results["predictions"][0] = 0
    with pytest.raises(ValueError):
        training_helpers.build_validation_review(validation_data, train_data, results, label2id)


@pytest.mark.parametrize("case", ["wrong_length", "non_integer", "missing_value", "zero", "negative", "retained_exceeds_original"])
def test_validation_review_rejects_invalid_token_statistics(validation_review_inputs, case):
    validation_data, train_data, results, label2id, encodings = validation_review_inputs
    if case == "wrong_length":
        encodings = encodings[:-1]
    elif case == "non_integer":
        encodings[0]["original_token_count"] = 12.5
    elif case == "missing_value":
        encodings[0]["original_token_count"] = None
    elif case == "zero":
        encodings[0]["retained_token_count"] = 0
    elif case == "negative":
        encodings[0]["original_token_count"] = -1
    else:
        encodings[0]["retained_token_count"] = encodings[0]["original_token_count"] + 1
    with pytest.raises(ValueError):
        training_helpers.build_validation_review(validation_data, train_data, results, label2id, encodings)


@pytest.mark.parametrize("label2id", [
    {}, {"alpha": 0}, {"alpha": 0, "beta": 2}, {"alpha": 0, "beta": 0},
    {"alpha": 0, "beta": True}, {"alpha": 0, "": 1}, {"alpha": 0, 3: 1},
])
def test_validation_review_rejects_invalid_label_mapping(validation_review_inputs, label2id):
    validation_data, train_data, results, _, _ = validation_review_inputs
    with pytest.raises(ValueError):
        training_helpers.build_validation_review(validation_data, train_data, results, label2id)


def _validation_review_notebook_namespace(validation_review_inputs):
    validation_data, train_data, results, label2id, encodings = validation_review_inputs
    dataset = EncodedNewsDataset(
        [{"input_ids": [0, 10, 2], "attention_mask": [1, 1, 1]} for _ in range(len(validation_data))],
        validation_data["label_id"].tolist(),
    )
    return {
        "pd": pd, "SequentialSampler": torch.utils.data.SequentialSampler,
        "build_validation_review": training_helpers.build_validation_review,
        "validation_data": validation_data, "train_data": train_data,
        "validation_results": results, "label2id": label2id,
        "validation_encodings": encodings, "validation_dataset": dataset,
        "validation_loader": DataLoader(dataset, batch_size=3, shuffle=False),
    }


def test_notebook_validation_review_order_guard_accepts_sequential_complete_dataset(validation_review_inputs):
    namespace = _validation_review_notebook_namespace(validation_review_inputs)
    exec(_notebook_cell_source("news05-review-order"), namespace)
    assert namespace["dataset_labels"] == validation_review_inputs[0]["label_id"].tolist()


@pytest.mark.parametrize("case", ["shuffled", "drop_last", "wrong_dataset", "wrong_record_labels"])
def test_notebook_validation_review_order_guard_rejects_unsafe_article_alignment(validation_review_inputs, case):
    namespace = _validation_review_notebook_namespace(validation_review_inputs)
    dataset = namespace["validation_dataset"]
    if case == "shuffled":
        namespace["validation_loader"] = DataLoader(dataset, batch_size=3, shuffle=True)
    elif case == "drop_last":
        namespace["validation_loader"] = DataLoader(dataset, batch_size=3, drop_last=True)
    elif case == "wrong_dataset":
        namespace["validation_loader"] = DataLoader(TinyClassificationDataset([0, 1, 2]))
    else:
        dataset.records[0]["labels"] = torch.tensor(2)
    with pytest.raises(ValueError):
        exec(_notebook_cell_source("news05-review-order"), namespace)


def test_notebook_validation_review_cells_never_evaluate_train_or_access_test():
    import ast

    for suffix in ("order", "build", "summary", "pairs", "articles", "save"):
        tree = ast.parse(_notebook_cell_source(f"news05-review-{suffix}"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not any(name.startswith("test") for name in names)
        assert not names.intersection({
            "evaluate_classifier", "train_one_epoch", "fit_classifier", "run_parameter_study",
            "model", "optimizer", "scheduler", "split_frames",
        })


def test_notebook_validation_review_builds_displays_and_saves_only_two_review_csvs(
    tmp_path, validation_review_inputs,
):
    namespace = _validation_review_notebook_namespace(validation_review_inputs)
    displayed = []
    namespace.update(RUN_REPORT_DIR=tmp_path, display=lambda table: displayed.append(table.copy()))
    for suffix in ("order", "build", "summary", "pairs", "articles", "save"):
        exec(_notebook_cell_source(f"news05-review-{suffix}"), namespace)
    assert len(displayed) == 3
    assert displayed[0]["news_type"].tolist() == ["absent", "alpha", "gamma", "beta"]
    assert {path.name for path in tmp_path.iterdir()} == {
        "validation_review_summary.csv", "validation_errors.csv",
    }
    for filename, key in (
        ("validation_review_summary.csv", "validation_review"),
        ("validation_errors.csv", "validation_errors"),
    ):
        path = tmp_path / filename
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
        saved = pd.read_csv(path, keep_default_na=False)
        expected = namespace[key]
        assert saved.columns.tolist() == expected.columns.tolist()
        assert len(saved) == len(expected)
    saved_errors = pd.read_csv(tmp_path / "validation_errors.csv")
    assert saved_errors["news_id"].tolist() == namespace["validation_errors"]["news_id"].tolist()
    assert len(saved_errors) == 5


@pytest.fixture
def saved_review_run(tmp_path):
    model_dir, report_dir = tmp_path / "models/run", tmp_path / "reports/run"
    for directory in (model_dir / "tokenizer", model_dir / "backbone", report_dir):
        directory.mkdir(parents=True)
    input_hashes = {
        "models/news_type_phobert_cnn/label_mapping.json": "a" * 64,
        **{f"data/splits/news_type/{name}.csv": str(index) * 64
           for index, name in enumerate(("train", "validation", "test"), 1)},
    }
    mapping = {"label_policy_version": "gold_price_industry_v1", "label2id": {"alpha": 0}}
    expected = {"max_length": 256, "head_ratio": 0.75, "kernel_sizes": [2, 3, 4, 5], "dropout": 0.5}
    resources = {"VnCoreNLP-1.2.jar": "4" * 64, "vi-vocab": "5" * 64, "wordsegmenter.rdr": "6" * 64}
    config = json.loads(json.dumps({
        **expected, "input_sha256": input_hashes, "label_mapping": mapping,
        "segmenter_sha256": resources, "best_epoch": 2, "best_validation_macro_f1": 0.8,
    }))
    (model_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    source_model = nn.Linear(2, 2)
    torch.save({
        "model_state_dict": source_model.state_dict(), "epoch": 2, "validation_macro_f1": 0.8,
    }, model_dir / "best_model.pt")
    history = pd.DataFrame({
        "epoch": [1, 2, 3], "train_loss": [1.0, 0.4, 0.2],
        "train_accuracy": [0.5, 0.8, 0.9], "train_macro_f1": [0.5, 0.8, 0.9],
        "validation_loss": [1.1, 0.5, 0.7], "validation_accuracy": [0.4, 0.8, 0.7],
        "validation_macro_f1": [0.4, 0.8, 0.7], "is_best": [True, True, False],
    })
    history.to_csv(report_dir / "training_history.csv", index=False)
    return SimpleNamespace(
        model_dir=model_dir, report_dir=report_dir, config=config, history=history,
        resources=resources, source_model=source_model,
        arguments={"input_sha256": input_hashes, "label_mapping": mapping, "expected_config": expected},
    )


def _load_review_run(run):
    return training_helpers.load_saved_run_metadata(run.model_dir, run.report_dir, **run.arguments)


def test_saved_run_metadata_preserves_artifacts_and_restores_history_attributes(saved_review_run):
    run = saved_review_run
    paths = [*run.model_dir.iterdir(), *run.report_dir.iterdir()]
    before = {path: path.read_bytes() for path in paths if path.is_file()}
    config, history = _load_review_run(run)
    assert config == run.config
    pd.testing.assert_frame_equal(history, run.history)
    assert history.attrs == {"best_epoch": 2, "best_validation_macro_f1": 0.8}
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("relative_path", [
    "model/best_model.pt", "model/run_config.json", "model/tokenizer", "model/backbone",
    "report/training_history.csv",
])
def test_saved_run_requires_all_original_artifacts(saved_review_run, relative_path):
    run = saved_review_run
    root_name, name = relative_path.split("/", 1)
    path = (run.model_dir if root_name == "model" else run.report_dir) / name
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()
    with pytest.raises(FileNotFoundError):
        _load_review_run(run)


@pytest.mark.parametrize("case", ["input_hash", "mapping", "policy", "max_length", "head_ratio", "kernel_sizes", "dropout"])
def test_saved_run_rejects_old_or_incompatible_metadata(saved_review_run, case):
    run = saved_review_run
    if case == "input_hash":
        run.config["input_sha256"]["data/splits/news_type/train.csv"] = "f" * 64
    elif case == "mapping":
        run.config["label_mapping"]["label2id"] = {"beta": 0}
    elif case == "policy":
        run.config["label_mapping"]["label_policy_version"] = "old"
    else:
        run.config[case] = {"max_length": 128, "head_ratio": 0.5, "kernel_sizes": [2, 3], "dropout": 0.3}[case]
    (run.model_dir / "run_config.json").write_text(json.dumps(run.config), encoding="utf-8")
    with pytest.raises(ValueError, match="match"):
        _load_review_run(run)


@pytest.mark.parametrize("case", ["missing_hash", "invalid_hash", "empty_mapping", "empty_config"])
def test_saved_run_rejects_incomplete_verification_inputs(saved_review_run, case):
    run = saved_review_run
    if case == "missing_hash":
        run.arguments["input_sha256"].pop("data/splits/news_type/test.csv")
    elif case == "invalid_hash":
        run.arguments["input_sha256"]["data/splits/news_type/test.csv"] = "not-a-sha256"
    elif case == "empty_mapping":
        run.arguments["label_mapping"] = {}
    else:
        run.arguments["expected_config"] = {}
    with pytest.raises(ValueError):
        _load_review_run(run)


@pytest.mark.parametrize("case", [
    "empty", "missing_column", "nan", "negative_loss", "score_above_one", "duplicate_epoch",
    "nonconsecutive_epoch", "best_epoch", "best_f1", "later_tied_best",
])
def test_saved_run_rejects_invalid_or_mismatched_training_history(saved_review_run, case):
    run = saved_review_run
    history = run.history.copy()
    if case == "empty":
        history = history.iloc[:0]
    elif case == "missing_column":
        history = history.drop(columns="train_accuracy")
    elif case == "nan":
        history.loc[0, "validation_macro_f1"] = np.nan
    elif case == "negative_loss":
        history.loc[0, "train_loss"] = -0.1
    elif case == "score_above_one":
        history.loc[0, "train_accuracy"] = 1.1
    elif case == "duplicate_epoch":
        history.loc[1, "epoch"] = 1
    elif case == "nonconsecutive_epoch":
        history.loc[2, "epoch"] = 4
    elif case == "best_epoch":
        run.config["best_epoch"] = 1
    elif case == "best_f1":
        run.config["best_validation_macro_f1"] = 0.7
    else:
        history.loc[0, "validation_macro_f1"] = 0.8
    history.to_csv(run.report_dir / "training_history.csv", index=False)
    (run.model_dir / "run_config.json").write_text(json.dumps(run.config), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_review_run(run)


@pytest.mark.parametrize("epoch, macro_f1", [(True, 0.8), (0, 0.8), (2.5, 0.8), (2, True), (2, float("inf")), (2, -0.1)])
def test_saved_run_rejects_invalid_best_metadata(saved_review_run, epoch, macro_f1):
    run = saved_review_run
    run.config.update(best_epoch=epoch, best_validation_macro_f1=macro_f1)
    (run.model_dir / "run_config.json").write_text(json.dumps(run.config), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_review_run(run)


def test_saved_checkpoint_restores_verified_weights_with_safe_cpu_load(saved_review_run, monkeypatch):
    run = saved_review_run
    destination = nn.Linear(2, 2)
    original_load = torch.load
    load_options = []

    def record_load(*args, **kwargs):
        load_options.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(training_helpers.torch, "load", record_load)
    before = (run.model_dir / "best_model.pt").read_bytes()
    metadata = training_helpers.restore_saved_checkpoint(destination, run.model_dir, run.config, "cpu")
    assert metadata == {"epoch": 2, "validation_macro_f1": 0.8}
    assert load_options == [{"map_location": "cpu", "weights_only": True}]
    assert not destination.training
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, run.source_model.state_dict()[name])
    assert (run.model_dir / "best_model.pt").read_bytes() == before


@pytest.mark.parametrize("case", ["epoch", "f1", "nan", "missing_state", "invalid_container"])
def test_saved_checkpoint_checks_metadata_before_changing_model(saved_review_run, case):
    run = saved_review_run
    checkpoint = {
        "model_state_dict": run.source_model.state_dict(), "epoch": 2, "validation_macro_f1": 0.8,
    }
    if case == "epoch":
        checkpoint["epoch"] = 1
    elif case == "f1":
        checkpoint["validation_macro_f1"] = 0.7
    elif case == "nan":
        checkpoint["validation_macro_f1"] = float("nan")
    elif case == "missing_state":
        checkpoint.pop("model_state_dict")
    else:
        checkpoint = []
    torch.save(checkpoint, run.model_dir / "best_model.pt")
    destination = nn.Linear(2, 2)
    before = {name: value.clone() for name, value in destination.state_dict().items()}
    with pytest.raises(ValueError):
        training_helpers.restore_saved_checkpoint(destination, run.model_dir, run.config, "cpu")
    assert destination.training
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_saved_checkpoint_requires_strict_state_dictionary(saved_review_run):
    run = saved_review_run
    checkpoint = {"model_state_dict": {}, "epoch": 2, "validation_macro_f1": 0.8}
    torch.save(checkpoint, run.model_dir / "best_model.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        training_helpers.restore_saved_checkpoint(nn.Linear(2, 2), run.model_dir, run.config, "cpu")


def test_saved_segmenter_verifies_exact_resource_fingerprints(saved_review_run):
    run = saved_review_run
    training_helpers.validate_saved_segmenter(run.config, run.resources.copy())
    for resources in ({}, {**run.resources, "extra": "7" * 64}, {**run.resources, "vi-vocab": "f" * 64}):
        with pytest.raises(ValueError, match="segmenter hashes"):
            training_helpers.validate_saved_segmenter(run.config, resources)
    with pytest.raises(ValueError):
        training_helpers.validate_saved_segmenter({}, run.resources)


def _revision_row_hash(row):
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@pytest.fixture
def approved_validation_revision(saved_review_run, tmp_path, monkeypatch):
    import src.cleaning_utils as cleaning_helpers
    from src.model_data_utils import MODEL_SPLIT_COLUMNS, build_phobert_input_text

    run = saved_review_run
    run_name = training_helpers.VALIDATION_REVISION_SOURCE_RUN
    model_dir, report_dir = run.model_dir.with_name(run_name), run.report_dir.with_name(run_name)
    run.model_dir.rename(model_dir)
    run.report_dir.rename(report_dir)
    run.model_dir, run.report_dir = model_dir, report_dir
    label_names = [
        "governance_capital", "financial_results", "business_operations", "legal_compliance",
        "industry", "stock_market", "mention_only",
    ]
    mapping = {"label_policy_version": "gold_price_industry_v1", "label2id": dict(zip(label_names, range(7)))}
    approved_ids = list(dict.fromkeys([
        *training_helpers._REVIEW_LABEL_CHANGES, *training_helpers._REVIEW_CONTENT_CHANGES,
        *sorted(training_helpers._REVIEW_PENDING_IDS), "unchanged-article",
    ]))
    rows = []
    for index, news_id in enumerate(approved_ids):
        label = training_helpers._REVIEW_LABEL_CHANGES.get(news_id, ("financial_results", ""))[0]
        body = (f"Article {index} reports ordinary operating information. " * 6).strip()
        mode = training_helpers._REVIEW_CONTENT_CHANGES.get(news_id)
        if mode == "prefix":
            body = "broken link prefix " + body
        elif mode == "suffix":
            body += " obsolete related-link tail"
        row = {name: f"{name}-{index}" for name in MODEL_SPLIT_COLUMNS}
        row.update(news_id=news_id, title=f"Title {index}", summary="", content=body,
                   news_type=label, label_id=str(mapping["label2id"][label]), split="validation")
        row["input_text"] = build_phobert_input_text(row["title"], row["summary"], body)
        rows.append(row)
    baseline = pd.DataFrame(rows, columns=MODEL_SPLIT_COLUMNS + ["split"])
    baseline_bytes = baseline.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig")
    run.config["input_sha256"]["data/splits/news_type/validation.csv"] = hashlib.sha256(baseline_bytes).hexdigest()
    run.config["label_mapping"] = mapping
    (model_dir / "run_config.json").write_text(json.dumps(run.config), encoding="utf-8")
    revised = baseline.copy(deep=True)
    synthetic_pins = {news_id: dict(record) for news_id, record in cleaning_helpers.REVIEWED_CONTENT_CORRECTIONS.items()}
    changes = []
    for position, row in baseline.iterrows():
        news_id = row["news_id"]
        fields = {}
        if news_id in training_helpers._REVIEW_LABEL_CHANGES:
            _, new_label = training_helpers._REVIEW_LABEL_CHANGES[news_id]
            for column, value in (("news_type", new_label), ("label_id", str(mapping["label2id"][new_label]))):
                fields[column] = {"before": row[column], "after": value}
                revised.at[position, column] = value
        if news_id in training_helpers._REVIEW_CONTENT_CHANGES:
            old_body = row["content"]
            mode = training_helpers._REVIEW_CONTENT_CHANGES[news_id]
            new_body = old_body.removeprefix("broken link prefix ") if mode == "prefix" else old_body.removesuffix(" obsolete related-link tail")
            new_input = build_phobert_input_text(row["title"], row["summary"], new_body)
            for column, value in (("content", new_body), ("input_text", new_input)):
                fields[column] = {"before": row[column], "after": value}
                revised.at[position, column] = value
        if fields:
            changes.append({"news_id": news_id, "before_row_sha256": _revision_row_hash(row.to_dict()),
                            "after_row_sha256": _revision_row_hash(revised.iloc[position].to_dict()), "fields": fields})
        if news_id in training_helpers._REVIEW_CONTENT_CHANGES:
            synthetic_pins[news_id].update(
                old_sha256=hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
                new_sha256=hashlib.sha256(revised.at[position, "content"].encode("utf-8")).hexdigest(),
            )
    monkeypatch.setattr(cleaning_helpers, "REVIEWED_CONTENT_CORRECTIONS", synthetic_pins)
    validation_path = tmp_path / "validation.csv"
    revised_bytes = revised.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig")
    validation_path.write_bytes(revised_bytes)
    current_hashes = {**run.config["input_sha256"], "data/splits/news_type/validation.csv": hashlib.sha256(revised_bytes).hexdigest(),
                      "models/news_type_phobert_cnn/label_mapping.json": "b" * 64}
    revision = {
        "revision_id": training_helpers.VALIDATION_REVISION_ID, "source_run": run_name,
        "baseline_run_config_sha256": hashlib.sha256((model_dir / "run_config.json").read_bytes()).hexdigest(),
        "baseline_label_policy_version": "gold_price_industry_v1",
        "revised_label_policy_version": "company_awards_listing_v2",
        "baseline_input_sha256": dict(run.config["input_sha256"]), "revised_input_sha256": current_hashes.copy(),
        "csv_format": {"encoding": "utf-8-sig", "lineterminator": "\r\n"}, "changes": changes,
        "pending_reviews": [{"news_id": news_id, "reason": "Pending human decision."}
                            for news_id in sorted(training_helpers._REVIEW_PENDING_IDS)],
    }
    return SimpleNamespace(run=run, baseline=baseline, revised=revised, revision=revision,
                           validation_path=validation_path, arguments={**run.arguments, "input_sha256": current_hashes,
                           "label_mapping": {**mapping, "label_policy_version": "company_awards_listing_v2"},
                           "validation_revision": revision, "validation_path": validation_path})


def _load_approved_revision(fixture):
    return training_helpers.load_saved_run_metadata(fixture.run.model_dir, fixture.run.report_dir, **fixture.arguments)


def _rewrite_revision_current_csv(fixture):
    payload = fixture.revised.to_csv(index=False, lineterminator="\r\n").encode("utf-8-sig")
    fixture.validation_path.write_bytes(payload)
    fingerprint = hashlib.sha256(payload).hexdigest()
    fixture.arguments["input_sha256"]["data/splits/news_type/validation.csv"] = fingerprint
    fixture.revision["revised_input_sha256"]["data/splits/news_type/validation.csv"] = fingerprint


def test_approved_validation_revision_replays_original_and_preserves_saved_artifacts(approved_validation_revision):
    fixture = approved_validation_revision
    paths = [*fixture.run.model_dir.iterdir(), *fixture.run.report_dir.iterdir(), fixture.validation_path]
    before = {path: path.read_bytes() for path in paths if path.is_file()}
    config, history = _load_approved_revision(fixture)
    assert config == fixture.run.config
    assert config["label_mapping"]["label_policy_version"] == "gold_price_industry_v1"
    assert history.attrs == {"best_epoch": 2, "best_validation_macro_f1": 0.8,
                             "evaluation_revision": "validation_review_20261025_v2"}
    assert len(fixture.revision["changes"]) == 8
    assert {path: path.read_bytes() for path in before} == before
    destination = nn.Linear(2, 2)
    assert training_helpers.restore_saved_checkpoint(destination, fixture.run.model_dir, config, "cpu") == {
        "epoch": 2, "validation_macro_f1": 0.8,
    }


def _confirm_review_articles(revision, count=2):
    decisions = revision["pending_reviews"][:count]
    revision["pending_reviews"] = revision["pending_reviews"][count:]
    revision["resolved_reviews"] = [
        {
            "news_id": item["news_id"], "news_type": "financial_results",
            "decision": "confirmed_unchanged", "reviewed_date": "2026-10-25",
            "reason": "Human review confirms earnings are the article's principal topic.",
        }
        for item in decisions
    ]


@pytest.mark.parametrize("confirmed_count", [1, 2])
def test_approved_validation_revision_accepts_confirmed_unchanged_reviews(
    approved_validation_revision, confirmed_count,
):
    fixture = approved_validation_revision
    original_data = fixture.validation_path.read_bytes()
    original_config = (fixture.run.model_dir / "run_config.json").read_bytes()
    _confirm_review_articles(fixture.revision, confirmed_count)
    config, history = _load_approved_revision(fixture)
    assert config == fixture.run.config
    assert history.attrs["evaluation_revision"] == training_helpers.VALIDATION_REVISION_ID
    assert len(fixture.revision["resolved_reviews"]) == confirmed_count
    assert len(fixture.revision["pending_reviews"]) == 2 - confirmed_count
    assert fixture.validation_path.read_bytes() == original_data
    assert (fixture.run.model_dir / "run_config.json").read_bytes() == original_config


@pytest.mark.parametrize("case", [
    "unaccounted", "duplicate", "overlap", "unknown", "wrong_label", "wrong_decision",
    "wrong_date", "missing_date", "empty_reason", "missing_reason", "invalid_resolved_list",
    "invalid_resolved_item", "changed_confirmed_label", "changed_confirmed_label_id",
])
def test_approved_validation_revision_rejects_invalid_review_resolution(
    approved_validation_revision, case,
):
    fixture = approved_validation_revision
    _confirm_review_articles(fixture.revision)
    revision = fixture.revision
    first = revision["resolved_reviews"][0]
    if case == "unaccounted":
        revision["resolved_reviews"] = []
    elif case == "duplicate":
        revision["resolved_reviews"][1] = first.copy()
    elif case == "overlap":
        revision["pending_reviews"] = [{"news_id": first["news_id"], "reason": "Pending again."}]
    elif case == "unknown":
        first["news_id"] = "not-approved"
    elif case == "wrong_label":
        first["news_type"] = "industry"
    elif case == "wrong_decision":
        first["decision"] = "changed_label"
    elif case == "wrong_date":
        first["reviewed_date"] = "invalid-date"
    elif case == "missing_date":
        first.pop("reviewed_date")
    elif case == "empty_reason":
        first["reason"] = " "
    elif case == "missing_reason":
        first.pop("reason")
    elif case == "invalid_resolved_list":
        revision["resolved_reviews"] = None
    elif case == "invalid_resolved_item":
        revision["resolved_reviews"][0] = "not-a-decision"
    else:
        column, value = (("news_type", "industry") if case == "changed_confirmed_label"
                         else ("label_id", "4"))
        fixture.revised.loc[fixture.revised["news_id"].eq(first["news_id"]), column] = value
        _rewrite_revision_current_csv(fixture)
    with pytest.raises(ValueError):
        _load_approved_revision(fixture)


def test_revised_validation_still_fails_without_explicit_approval_manifest(approved_validation_revision):
    fixture = approved_validation_revision
    fixture.arguments.pop("validation_revision")
    with pytest.raises(ValueError, match="input hashes"):
        _load_approved_revision(fixture)


@pytest.mark.parametrize("case", [
    "revision_id", "source_run", "config_binding", "baseline_hash", "revised_hash", "train_hash", "test_hash",
    "baseline_policy", "new_policy", "mapping_ids", "csv_encoding", "csv_newline", "pending_missing",
    "pending_changed", "pending_empty_reason", "duplicate_change", "missing_change", "unapproved_id",
    "unapproved_field", "before_row_hash", "after_row_hash", "before_field", "after_field", "no_op_field",
    "numeric_label_id", "missing_validation_path",
])
def test_approved_validation_revision_rejects_manifest_tampering(approved_validation_revision, case):
    fixture = approved_validation_revision
    revision = fixture.revision
    change = revision["changes"][0]
    if case == "revision_id":
        revision["revision_id"] = "unapproved"
    elif case == "source_run":
        revision["source_run"] = "old-study"
    elif case == "config_binding":
        revision["baseline_run_config_sha256"] = "f" * 64
    elif case == "baseline_hash":
        revision["baseline_input_sha256"]["data/splits/news_type/validation.csv"] = "f" * 64
    elif case == "revised_hash":
        revision["revised_input_sha256"]["data/splits/news_type/validation.csv"] = "f" * 64
    elif case in {"train_hash", "test_hash"}:
        name = case.removesuffix("_hash")
        fixture.arguments["input_sha256"][f"data/splits/news_type/{name}.csv"] = "f" * 64
        revision["revised_input_sha256"][f"data/splits/news_type/{name}.csv"] = "f" * 64
    elif case == "baseline_policy":
        revision["baseline_label_policy_version"] = "unknown"
    elif case == "new_policy":
        revision["revised_label_policy_version"] = "unknown"
    elif case == "mapping_ids":
        fixture.arguments["label_mapping"]["label2id"] = {"changed": 0}
    elif case == "csv_encoding":
        revision["csv_format"]["encoding"] = "latin-1"
    elif case == "csv_newline":
        revision["csv_format"]["lineterminator"] = "\n"
    elif case == "pending_missing":
        revision["pending_reviews"].pop()
    elif case == "pending_changed":
        revision["pending_reviews"][0]["news_id"] = "wrong"
    elif case == "pending_empty_reason":
        revision["pending_reviews"][0]["reason"] = " "
    elif case == "duplicate_change":
        revision["changes"][-1] = change.copy()
    elif case == "missing_change":
        revision["changes"].pop()
    elif case == "unapproved_id":
        change["news_id"] = "unchanged-article"
    elif case == "unapproved_field":
        change["fields"]["title"] = {"before": "before", "after": "after"}
    elif case == "before_row_hash":
        change["before_row_sha256"] = "f" * 64
    elif case == "after_row_hash":
        change["after_row_sha256"] = "f" * 64
    elif case == "before_field":
        change["fields"]["news_type"]["before"] = "industry"
    elif case == "after_field":
        change["fields"]["news_type"]["after"] = "industry"
    elif case == "no_op_field":
        change["fields"]["news_type"]["before"] = change["fields"]["news_type"]["after"]
    elif case == "numeric_label_id":
        change["fields"]["label_id"]["after"] = 2
    else:
        fixture.arguments["validation_path"] = None
    with pytest.raises(ValueError):
        _load_approved_revision(fixture)


@pytest.mark.parametrize("case", [
    "unchanged_title", "pending_label", "row_order", "removed_pending", "added_article", "immutable_source",
    "invented_content", "extra_boundary_trim", "wrong_rebuilt_text", "wrong_approved_label", "missing_current_hash",
])
def test_approved_validation_revision_rejects_unapproved_current_data(approved_validation_revision, case):
    from src.model_data_utils import build_phobert_input_text

    fixture = approved_validation_revision
    changed_id = next(iter(training_helpers._REVIEW_CONTENT_CHANGES))
    position = fixture.revised.index[fixture.revised["news_id"].eq(changed_id)][0]
    if case == "unchanged_title":
        fixture.revised.loc[fixture.revised["news_id"].eq("unchanged-article"), "title"] = "Edited title"
    elif case == "pending_label":
        fixture.revised.loc[fixture.revised["news_id"].isin(training_helpers._REVIEW_PENDING_IDS), "news_type"] = "industry"
    elif case == "row_order":
        fixture.revised = fixture.revised.iloc[::-1].reset_index(drop=True)
    elif case == "removed_pending":
        fixture.revised = fixture.revised.loc[~fixture.revised["news_id"].isin(training_helpers._REVIEW_PENDING_IDS)].copy()
    elif case == "added_article":
        extra = fixture.revised.iloc[-1].copy()
        extra["news_id"] = "extra-article"
        fixture.revised = pd.concat([fixture.revised, extra.to_frame().T], ignore_index=True)
    elif case == "immutable_source":
        fixture.revised.at[position, "source"] = "Different source"
    elif case in {"invented_content", "extra_boundary_trim"}:
        fixture.revised.at[position, "content"] = (
            "Invented information that did not appear in the original article. " * 6 if case == "invented_content"
            else fixture.revised.at[position, "content"][8:]
        )
        record = fixture.revised.loc[position]
        fixture.revised.at[position, "input_text"] = build_phobert_input_text(record["title"], record["summary"], record["content"])
    elif case == "wrong_rebuilt_text":
        fixture.revised.at[position, "input_text"] = "Another input"
    elif case == "wrong_approved_label":
        fixture.revised.at[0, "news_type"] = "industry"
        fixture.revised.at[0, "label_id"] = "4"
    else:
        fixture.validation_path.write_bytes(b"unexpected bytes")
    if case != "missing_current_hash":
        _rewrite_revision_current_csv(fixture)
    if case in {"immutable_source", "invented_content", "extra_boundary_trim", "wrong_rebuilt_text", "wrong_approved_label"}:
        for change in fixture.revision["changes"]:
            row = fixture.revised.loc[fixture.revised["news_id"].eq(change["news_id"])].iloc[0].to_dict()
            change["after_row_sha256"] = _revision_row_hash(row)
            for column, values in change["fields"].items():
                values["after"] = row[column]
    with pytest.raises(ValueError):
        _load_approved_revision(fixture)


@pytest.fixture
def exported_test_run(tmp_path):
    """Small exported reports, not a trained or downloaded production checkpoint."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report, f1_score,
    )

    model_dir = tmp_path / "models" / "selected-run"
    report_dir = tmp_path / "reports" / "validation-revision"
    (model_dir / "tokenizer").mkdir(parents=True)
    (model_dir / "backbone").mkdir()
    report_dir.mkdir(parents=True)
    (model_dir / "best_model.pt").write_bytes(b"offline checkpoint identity fixture")
    for relative, content in {
        "tokenizer/tokenizer_config.json": '{"tokenizer_class":"PhoBertTokenizer"}',
        "tokenizer/vocab.txt": "fixture 1\n",
        "tokenizer/bpe.codes": "# Offline fixture\n",
        "backbone/config.json": '{"model_type":"roberta","hidden_size":8}',
    }.items():
        (model_dir / relative).write_text(content, encoding="utf-8")
    labels = {"alpha": 0, "beta": 1, "gamma": 2}
    mapping = {
        "label_policy_version": "fixture-v1", "label2id": labels,
        "id2label": {str(value): label for label, value in labels.items()},
    }
    input_hashes = {
        "models/news_type_phobert_cnn/label_mapping.json": "a" * 64,
        **{f"data/splits/news_type/{name}.csv": str(index) * 64
           for index, name in enumerate(("train", "validation", "test"), start=1)},
    }
    resources = {"segmenter-resource": "b" * 64}
    config = {
        "input_sha256": input_hashes, "label_mapping": mapping, "num_classes": 3,
        "max_length": 256, "head_ratio": 0.75, "segmenter_sha256": resources,
        "best_epoch": 2, "best_validation_macro_f1": 0.8,
    }
    (model_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    truth = np.array([0, 0, 1, 1, 2, 2])
    guessed = np.array([0, 1, 1, 2, 2, 0])
    test_data = pd.DataFrame({
        "news_id": [f"test-{index}" for index in range(len(truth))],
        "source": "Offline", "news_type": [list(labels)[value] for value in truth],
        "label_id": truth, "split": "test",
    })
    predictions = test_data[["news_id", "source", "news_type"]].copy()
    predictions["predicted_news_type"] = [list(labels)[value] for value in guessed]
    predictions["predicted_probability"] = 0.7
    predictions["is_correct"] = truth == guessed
    predictions.to_csv(report_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "loss": 0.6, "accuracy": float(accuracy_score(truth, guessed)),
        "macro_f1": float(f1_score(truth, guessed, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, guessed, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, guessed)),
    }
    (report_dir / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    class_report = pd.DataFrame(classification_report(
        truth, guessed, labels=[0, 1, 2], target_names=list(labels), output_dict=True, zero_division=0,
    )).T
    class_report.to_csv(report_dir / "test_classification_report.csv", encoding="utf-8-sig")
    return SimpleNamespace(
        model_dir=model_dir, report_dir=report_dir, config=config, metrics=metrics,
        arguments={
            "input_sha256": input_hashes, "label_mapping": mapping,
            "validation_revision_sha256": None, "segmenter_sha256": resources,
            "max_length": 256, "head_ratio": 0.75, "test_data": test_data,
        },
    )


def _build_exported_test_manifest(run):
    return training_helpers.build_test_evaluation_manifest(run.model_dir, run.report_dir, **run.arguments)


def _save_exported_test_manifest(run):
    manifest = _build_exported_test_manifest(run)
    (run.report_dir / "test_evaluation_config.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_test_provenance_binds_real_exported_artifacts_without_writes(exported_test_run):
    run = exported_test_run
    protected = {path: path.read_bytes() for directory in (run.model_dir, run.report_dir)
                 for path in directory.rglob("*") if path.is_file()}
    manifest = _build_exported_test_manifest(run)
    assert not (run.report_dir / "test_evaluation_config.json").exists()
    assert {path: path.read_bytes() for path in protected} == protected
    assert manifest["source_run"] == "selected-run"
    assert manifest["evaluated_split"] == "test" and manifest["test_evaluated"] is True
    assert manifest["test_count"] == 6 and manifest["metrics"] == run.metrics
    assert set(manifest["artifact_sha256"]) == {
        "test_metrics.json", "test_predictions.csv", "test_classification_report.csv",
    }
    assert manifest["checkpoint_sha256"] == hashlib.sha256(
        (run.model_dir / "best_model.pt").read_bytes()
    ).hexdigest()
    assert "tokenizer/vocab.txt" in manifest["model_assets_sha256"]
    assert "backbone/config.json" in manifest["model_assets_sha256"]
    manifest["label_mapping"]["label2id"]["alpha"] = 99
    assert run.arguments["label_mapping"]["label2id"]["alpha"] == 0


def test_test_provenance_verification_preserves_all_saved_files(exported_test_run):
    run = exported_test_run
    manifest = _save_exported_test_manifest(run)
    protected = {path: path.read_bytes() for directory in (run.model_dir, run.report_dir)
                 for path in directory.rglob("*") if path.is_file()}
    observed = training_helpers.verify_test_evaluation_manifest(run.model_dir, run.report_dir, **run.arguments)
    assert observed == manifest
    assert {path: path.read_bytes() for path in protected} == protected


def test_missing_test_provenance_fails_before_checkpoint_hash_or_model_work(exported_test_run, monkeypatch):
    run = exported_test_run

    def forbidden(*args, **kwargs):
        raise AssertionError("No checkpoint work is allowed before the missing-test gate.")

    monkeypatch.setattr(training_helpers, "_artifact_sha256", forbidden)
    monkeypatch.setattr(training_helpers, "build_test_evaluation_manifest", forbidden)
    with pytest.raises(FileNotFoundError, match="notebook 05 section 14"):
        training_helpers.verify_test_evaluation_manifest(run.model_dir, run.report_dir, **run.arguments)


@pytest.mark.parametrize("relative", [
    "model/best_model.pt", "model/run_config.json", "model/tokenizer/tokenizer_config.json",
    "model/backbone/config.json", "report/test_metrics.json", "report/test_predictions.csv",
    "report/test_classification_report.csv",
])
def test_test_provenance_requires_actual_saved_assets_and_all_reports(exported_test_run, relative):
    run = exported_test_run
    kind, name = relative.split("/", 1)
    ((run.model_dir if kind == "model" else run.report_dir) / name).unlink()
    with pytest.raises(FileNotFoundError):
        _build_exported_test_manifest(run)


@pytest.mark.parametrize("field", [
    "source_run", "checkpoint_sha256", "run_config_sha256", "model_assets_sha256",
    "input_sha256", "label_mapping", "segmenter_sha256", "validation_revision_sha256",
    "max_length", "head_ratio", "test_count", "checkpoint_epoch", "metrics", "artifact_sha256",
    "evaluated_split", "test_evaluated", "schema_version", "unknown_metadata",
])
def test_test_provenance_rejects_wrong_or_tampered_manifest(exported_test_run, field):
    run = exported_test_run
    manifest = _save_exported_test_manifest(run)
    if field == "test_evaluated":
        manifest[field] = False
    elif field == "schema_version":
        manifest[field] = True  # bool must not be accepted as the integer version 1.
    elif field in {"max_length", "head_ratio", "test_count", "checkpoint_epoch"}:
        manifest[field] += 1
    else:
        manifest[field] = "changed"
    (run.report_dir / "test_evaluation_config.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        training_helpers.verify_test_evaluation_manifest(run.model_dir, run.report_dir, **run.arguments)


@pytest.mark.parametrize("relative", [
    "best_model.pt", "tokenizer/tokenizer_config.json", "tokenizer/vocab.txt",
    "tokenizer/bpe.codes", "backbone/config.json",
])
def test_test_provenance_rejects_changed_model_or_tokenizer_after_test(exported_test_run, relative):
    run = exported_test_run
    _save_exported_test_manifest(run)
    path = run.model_dir / relative
    path.write_bytes(path.read_bytes() + b" changed after test")
    with pytest.raises(ValueError, match="provenance"):
        training_helpers.verify_test_evaluation_manifest(run.model_dir, run.report_dir, **run.arguments)


@pytest.mark.parametrize("case", [
    "row_order", "removed_article", "duplicate_id", "wrong_source", "wrong_truth", "unknown_prediction",
    "wrong_correct_flag", "nan_score", "infinite_score", "negative_score", "score_above_one", "missing_column",
])
def test_test_provenance_rejects_invalid_or_misaligned_prediction_exports(exported_test_run, case):
    run = exported_test_run
    path = run.report_dir / "test_predictions.csv"
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if case == "row_order":
        data = data.iloc[::-1]
    elif case == "removed_article":
        data = data.iloc[:-1]
    elif case == "duplicate_id":
        data.loc[1, "news_id"] = data.loc[0, "news_id"]
    elif case == "wrong_source":
        data.loc[0, "source"] = "Another source"
    elif case == "wrong_truth":
        data.loc[0, "news_type"] = "gamma"
    elif case == "unknown_prediction":
        data.loc[0, "predicted_news_type"] = "unknown"
    elif case == "wrong_correct_flag":
        data.loc[0, "is_correct"] = "False"
    elif case == "missing_column":
        data = data.drop(columns="is_correct")
    else:
        data.loc[0, "predicted_probability"] = {
            "nan_score": "NaN", "infinite_score": "inf", "negative_score": "-0.1", "score_above_one": "1.1",
        }[case]
    data.to_csv(path, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError):
        _build_exported_test_manifest(run)


@pytest.mark.parametrize("case", [
    "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "negative_loss", "nan_loss",
    "bool_score", "missing_metric", "per_class_report",
])
def test_test_provenance_recomputes_scores_instead_of_trusting_report_files(exported_test_run, case):
    run = exported_test_run
    if case == "per_class_report":
        path = run.report_dir / "test_classification_report.csv"
        data = pd.read_csv(path, index_col=0)
        data.loc["alpha", "f1-score"] = 0.99
        data.to_csv(path, encoding="utf-8-sig")
    else:
        metrics = dict(run.metrics)
        if case == "negative_loss":
            metrics["loss"] = -0.1
        elif case == "nan_loss":
            metrics["loss"] = float("nan")
        elif case == "bool_score":
            metrics["accuracy"] = True
        elif case == "missing_metric":
            metrics.pop("weighted_f1")
        else:
            metrics[case] = 0.9
        (run.report_dir / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError):
        _build_exported_test_manifest(run)


@pytest.mark.parametrize("case", [
    "missing_hash", "bad_hash", "train_hash", "test_hash", "mapping", "class_count", "max_length",
    "head_ratio", "bool_length", "nan_ratio", "segmenter", "bad_segmenter_hash", "bad_revision_hash",
])
def test_test_provenance_rejects_incompatible_model_data_or_preprocessing(exported_test_run, case):
    run = exported_test_run
    arguments = run.arguments
    if case == "missing_hash":
        arguments["input_sha256"].pop("data/splits/news_type/validation.csv")
    elif case == "bad_hash":
        arguments["input_sha256"]["data/splits/news_type/validation.csv"] = "not-a-digest"
    elif case in {"train_hash", "test_hash"}:
        arguments["input_sha256"][f"data/splits/news_type/{case.removesuffix('_hash')}.csv"] = "f" * 64
    elif case == "mapping":
        arguments["label_mapping"]["label2id"] = {"alpha": 1, "beta": 0, "gamma": 2}
    elif case == "class_count":
        run.config["num_classes"] = 2
        (run.model_dir / "run_config.json").write_text(json.dumps(run.config), encoding="utf-8")
    elif case == "max_length":
        arguments["max_length"] = 128
    elif case == "head_ratio":
        arguments["head_ratio"] = 0.5
    elif case == "bool_length":
        arguments["max_length"] = True
    elif case == "nan_ratio":
        arguments["head_ratio"] = float("nan")
    elif case == "segmenter":
        arguments["segmenter_sha256"] = {"segmenter-resource": "f" * 64}
    elif case == "bad_segmenter_hash":
        arguments["segmenter_sha256"] = {"segmenter-resource": "invalid"}
    else:
        arguments["validation_revision_sha256"] = "invalid"
    with pytest.raises(ValueError):
        _build_exported_test_manifest(run)


def test_test_provenance_requires_revision_for_changed_validation_and_policy(exported_test_run):
    run = exported_test_run
    run.arguments["input_sha256"]["data/splits/news_type/validation.csv"] = "f" * 64
    run.arguments["input_sha256"]["models/news_type_phobert_cnn/label_mapping.json"] = "e" * 64
    run.arguments["label_mapping"]["label_policy_version"] = "reviewed-v2"
    with pytest.raises(ValueError, match="approved revision"):
        _build_exported_test_manifest(run)
    run.arguments["validation_revision_sha256"] = "d" * 64
    manifest = _save_exported_test_manifest(run)
    assert manifest["validation_revision_sha256"] == "d" * 64
    assert training_helpers.verify_test_evaluation_manifest(
        run.model_dir, run.report_dir, **run.arguments,
    ) == manifest


@pytest.mark.parametrize("case", ["empty", "wrong_split", "duplicate", "wrong_label_id", "missing_source"])
def test_test_provenance_rejects_invalid_ground_truth_table(exported_test_run, case):
    run = exported_test_run
    data = run.arguments["test_data"].copy()
    if case == "empty":
        data = data.iloc[:0]
    elif case == "wrong_split":
        data["split"] = "validation"
    elif case == "duplicate":
        data.loc[1, "news_id"] = data.loc[0, "news_id"]
    elif case == "wrong_label_id":
        data.loc[0, "label_id"] = 1
    else:
        data = data.drop(columns="source")
    run.arguments["test_data"] = data
    with pytest.raises(ValueError):
        _build_exported_test_manifest(run)


def test_test_provenance_accepts_string_ground_truth_ids_without_mutating_input(exported_test_run):
    run = exported_test_run
    run.arguments["test_data"]["label_id"] = run.arguments["test_data"]["label_id"].astype(str)
    before = run.arguments["test_data"].copy(deep=True)
    assert _build_exported_test_manifest(run)["test_count"] == len(before)
    pd.testing.assert_frame_equal(run.arguments["test_data"], before)


def test_test_provenance_does_not_allow_artifact_path_escape(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        training_helpers._scoped_test_file(tmp_path / "report", "../unrelated.json")
