"""Offline inference checks: human labels, lineage, row identity and no training."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.annotation_utils import ANNOTATION_COLUMNS, NEWS_COLUMNS
from src.news_type_inference import (
    OUTPUT_COLUMNS,
    PREDICTION_COLUMNS,
    UnlabeledNewsDataset,
    build_inference_plan,
    combine_news_type_predictions,
    predict_news_types,
)
from src.phobert_cnn import PhoBertCNN, encode_news_texts


LABELS = {"financial_results": 0, "industry": 1}


def article(news_id, **updates):
    values = {
        "news_id": news_id, "source": "Fixture", "url": f"https://fixture.test/{news_id}",
        "title": f"Title {news_id}", "summary": "A useful summary.",
        "content": f"Full article body for {news_id}.",
        "published_date": "2025-01-03", "tickers": "MWG",
    }
    values.update(updates)
    return values


def annotation(news_id, label="financial_results", status="labeled", **updates):
    return {
        **article(news_id), "news_type": label, "annotation_status": status,
        "annotator_note": "", **updates,
    }


@pytest.fixture
def inputs():
    return {
        "clean_news": pd.DataFrame([article("manual"), article("target"), article("hold")], columns=NEWS_COLUMNS),
        "annotations": pd.DataFrame([
            annotation("manual"), annotation("hold", "industry", "review_required"),
        ], columns=NEWS_COLUMNS + ANNOTATION_COLUMNS),
        "excluded_records": pd.DataFrame(columns=["news_id", "exclusion_reason", "content"]),
        "duplicate_merge_log": pd.DataFrame(columns=["kept_news_id", "merged_news_ids"]),
        "label2id": LABELS,
    }


def test_plan_preserves_manual_decisions_and_holds_unresolved_articles(inputs):
    before = {name: value.copy(deep=True) for name, value in inputs.items() if isinstance(value, pd.DataFrame)}
    plan, targets, audit = build_inference_plan(**inputs)
    assert plan.columns.tolist() == OUTPUT_COLUMNS
    assert plan.to_dict("records") == [
        {"news_id": "manual", "news_type": "financial_results", "label_source": "manual", "prediction_score": ""},
        {"news_id": "target", "news_type": "", "label_source": "unlabeled", "prediction_score": ""},
        {"news_id": "hold", "news_type": "", "label_source": "review_required", "prediction_score": ""},
    ]
    assert targets.columns.tolist() == ["news_id", "input_text"]
    assert targets["news_id"].tolist() == ["target"]
    assert targets.iloc[0]["input_text"] == "Title target. A useful summary. Full article body for target."
    assert audit["training_eligible"].tolist() == [True, False]
    for name, frame in before.items():
        pd.testing.assert_frame_equal(inputs[name], frame)


def test_plan_resolves_deleted_annotation_alias_through_duplicate_log(inputs):
    inputs["annotations"].loc[0, "news_id"] = "old-manual"
    inputs["annotations"].loc[0, "url"] = "https://fixture.test/old-manual"
    inputs["duplicate_merge_log"] = pd.DataFrame([
        {"kept_news_id": "manual", "merged_news_ids": "old-manual|manual"},
    ])
    plan, targets, audit = build_inference_plan(**inputs)
    assert plan.iloc[0]["label_source"] == "manual"
    assert audit.iloc[0]["canonical_news_id"] == "manual"
    assert targets["news_id"].tolist() == ["target"]


def test_plan_resolves_canonical_url_recovery_without_relabeling(inputs):
    inputs["annotations"].loc[0, "news_id"] = "old-manual"
    plan, _, audit = build_inference_plan(**inputs)
    assert plan.iloc[0]["news_type"] == "financial_results"
    assert audit.iloc[0]["canonical_news_id"] == "manual"


def test_finalized_alias_does_not_hide_an_unresolved_human_decision(inputs):
    duplicate = annotation("alias", label="industry", status="review_required")
    inputs["annotations"] = pd.concat([inputs["annotations"], pd.DataFrame([duplicate])], ignore_index=True)
    inputs["duplicate_merge_log"] = pd.DataFrame([
        {"kept_news_id": "manual", "merged_news_ids": "manual|alias"},
    ])
    plan, targets, _ = build_inference_plan(**inputs)
    assert plan.iloc[0]["label_source"] == "review_required"
    assert plan.iloc[0]["news_type"] == ""
    assert targets["news_id"].tolist() == ["target"]


def test_duplicate_log_cannot_redirect_an_existing_clean_article_or_its_manual_label(inputs):
    before = {name: frame.copy(deep=True) for name, frame in inputs.items() if isinstance(frame, pd.DataFrame)}
    inputs["duplicate_merge_log"] = pd.DataFrame([
        {"kept_news_id": "target", "merged_news_ids": "manual|target"},
    ])
    with pytest.raises(ValueError, match="cannot redirect"):
        build_inference_plan(**inputs)
    for name in ("clean_news", "annotations", "excluded_records"):
        pd.testing.assert_frame_equal(inputs[name], before[name])


@pytest.mark.parametrize("other_label, expected_source", [("financial_results", "manual"), ("industry", "review_required")])
def test_duplicate_human_decisions_never_get_replaced_by_model(inputs, other_label, expected_source):
    duplicate = annotation("alias", label=other_label)
    inputs["annotations"] = pd.concat([inputs["annotations"], pd.DataFrame([duplicate])], ignore_index=True)
    inputs["duplicate_merge_log"] = pd.DataFrame([
        {"kept_news_id": "manual", "merged_news_ids": "manual|alias"},
    ])
    plan, targets, _ = build_inference_plan(**inputs)
    assert plan.iloc[0]["label_source"] == expected_source
    assert targets["news_id"].tolist() == ["target"]


@pytest.mark.parametrize("label, status, note", [
    ("", "review_required", ""), ("industry", "review_required", ""),
    ("financial_results", "excluded", "Do not use"), ("industry", "", ""),
    ("", "", "Human review started"),
])
def test_unfinished_or_excluded_human_work_stays_on_hold(inputs, label, status, note):
    inputs["annotations"].loc[1, ANNOTATION_COLUMNS] = [label, status, note]
    plan, targets, _ = build_inference_plan(**inputs)
    assert plan.iloc[2]["label_source"] == "review_required"
    assert plan.iloc[2]["news_type"] == ""
    assert targets["news_id"].tolist() == ["target"]


def test_blank_unstarted_annotation_is_genuinely_unlabeled(inputs):
    inputs["annotations"].loc[1, ANNOTATION_COLUMNS] = ["", "", ""]
    plan, targets, _ = build_inference_plan(**inputs)
    assert plan.iloc[2]["label_source"] == "unlabeled"
    assert targets["news_id"].tolist() == ["target", "hold"]


def test_no_annotations_makes_all_clean_articles_targets(inputs):
    inputs["annotations"] = inputs["annotations"].iloc[:0]
    plan, targets, audit = build_inference_plan(**inputs)
    assert plan["label_source"].eq("unlabeled").all()
    assert targets["news_id"].tolist() == inputs["clean_news"]["news_id"].tolist()
    assert audit.empty and "training_eligible" in audit


def test_legacy_macro_manual_label_uses_existing_industry_mapping(inputs):
    inputs["annotations"].loc[0, "news_type"] = "macro"
    plan, _, _ = build_inference_plan(**inputs)
    assert plan.iloc[0]["news_type"] == "industry"


def test_finalized_annotation_outside_clean_is_not_reintroduced(inputs):
    outside = annotation("excluded-original")
    inputs["annotations"] = pd.concat([inputs["annotations"], pd.DataFrame([outside])], ignore_index=True)
    inputs["excluded_records"] = pd.DataFrame([
        {"news_id": "excluded-original", "exclusion_reason": "CONTENT_QUALITY:TITLE_ONLY", "content": "bad"},
    ])
    plan, targets, audit = build_inference_plan(**inputs)
    assert len(plan) == 3 and targets["news_id"].tolist() == ["target"]
    assert not audit.iloc[-1]["training_eligible"]


@pytest.mark.parametrize("case", ["unknown_label", "empty_label", "unknown_status", "duplicate_news_id", "duplicate_index", "duplicate_url", "bad_clean_schema", "unresolved_lineage", "conflicting_lineage"])
def test_plan_fails_on_ambiguous_or_invalid_inputs(inputs, case):
    if case == "unknown_label":
        inputs["annotations"].loc[0, "news_type"] = "not-a-label"
    elif case == "empty_label":
        inputs["annotations"].loc[0, "news_type"] = ""
    elif case == "unknown_status":
        inputs["annotations"].loc[0, "annotation_status"] = "finished-ish"
    elif case == "duplicate_news_id":
        inputs["clean_news"].loc[1, "news_id"] = "manual"
    elif case == "duplicate_index":
        inputs["annotations"].index = [0, 0]
    elif case == "duplicate_url":
        inputs["clean_news"].loc[1, "url"] = inputs["clean_news"].loc[0, "url"]
    elif case == "bad_clean_schema":
        inputs["clean_news"] = inputs["clean_news"].drop(columns="summary")
    elif case == "unresolved_lineage":
        inputs["annotations"].loc[0, "news_id"] = "unknown"
        inputs["annotations"].loc[0, "url"] = "https://fixture.test/unknown"
    else:
        inputs["duplicate_merge_log"] = pd.DataFrame([
            {"kept_news_id": "manual", "merged_news_ids": "alias"},
            {"kept_news_id": "target", "merged_news_ids": "alias"},
        ])
    with pytest.raises(ValueError):
        build_inference_plan(**inputs)


def test_merge_fills_only_targets_and_preserves_original_clean_order(inputs):
    inputs["clean_news"] = pd.concat([
        inputs["clean_news"], pd.DataFrame([article("target2")]),
    ], ignore_index=True)
    plan, _, _ = build_inference_plan(**inputs)
    before = plan.copy(deep=True)
    predictions = pd.DataFrame([
        {"news_id": "target2", "news_type": "industry", "prediction_score": 0.8},
        {"news_id": "target", "news_type": "financial_results", "prediction_score": 0.7},
    ], columns=PREDICTION_COLUMNS)
    result = combine_news_type_predictions(plan, predictions, LABELS)
    assert result["news_id"].tolist() == ["manual", "target", "hold", "target2"]
    assert result["news_type"].tolist() == ["financial_results", "financial_results", "", "industry"]
    assert result["label_source"].tolist() == ["manual", "model", "review_required", "model"]
    assert result["prediction_score"].tolist() == ["", 0.7, "", 0.8]
    pd.testing.assert_frame_equal(plan, before)


@pytest.mark.parametrize("case", ["missing", "manual", "hold", "duplicate", "unknown_label", "nan", "inf", "negative", "above_one", "bool", "manual_score", "unconfirmed_label"])
def test_merge_rejects_incomplete_predictions_and_human_overwrites(inputs, case):
    plan, _, _ = build_inference_plan(**inputs)
    predictions = pd.DataFrame([
        {"news_id": "target", "news_type": "industry", "prediction_score": 0.8},
    ], columns=PREDICTION_COLUMNS)
    if case == "missing":
        predictions = predictions.iloc[:0]
    elif case in {"manual", "hold"}:
        predictions.loc[0, "news_id"] = case
    elif case == "duplicate":
        predictions = pd.concat([predictions, predictions], ignore_index=True)
    elif case == "unknown_label":
        predictions.loc[0, "news_type"] = "unsupported"
    elif case == "manual_score":
        plan.loc[0, "prediction_score"] = 1.0
    elif case == "unconfirmed_label":
        plan.loc[2, "news_type"] = "industry"
    else:
        values = {"nan": np.nan, "inf": np.inf, "negative": -0.1, "above_one": 1.1, "bool": True}
        predictions["prediction_score"] = pd.Series([values[case]], dtype=object)
    with pytest.raises(ValueError):
        combine_news_type_predictions(plan, predictions, LABELS)


def test_merge_empty_prediction_pool_keeps_manual_and_review_rows(inputs):
    inputs["clean_news"] = inputs["clean_news"].iloc[[0, 2]]
    plan, targets, _ = build_inference_plan(**inputs)
    assert targets.empty
    result = combine_news_type_predictions(plan, pd.DataFrame(columns=PREDICTION_COLUMNS), LABELS)
    pd.testing.assert_frame_equal(plan, result)


def encodings(ids):
    return [
        {"news_id": news_id, "input_ids": [index + 2, 1, 1], "attention_mask": [1, 0, 0]}
        for index, news_id in enumerate(ids)
    ]


class TinyInferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 2)
        with torch.no_grad():
            self.embedding.weight.fill_(0)
            self.embedding.weight[2] = torch.tensor([3.0, -1.0])
            self.embedding.weight[3] = torch.tensor([-1.0, 3.0])
            self.embedding.weight[4] = torch.tensor([2.0, 0.0])
        self.events = []

    def forward(self, input_ids, attention_mask):
        self.events.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return self.embedding(input_ids[:, 0])


def test_prediction_is_sequential_eval_only_and_does_not_change_model_weights():
    model = TinyInferenceModel().train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    targets = pd.DataFrame({"news_id": ["a", "b", "c"], "input_text": ["One", "Two", "Three"]})
    encoded = encodings(targets["news_id"])
    encoded_before = copy.deepcopy(encoded)
    result = predict_news_types(model, targets, encoded, "cpu", LABELS, batch_size=2)
    assert result.columns.tolist() == PREDICTION_COLUMNS
    assert result["news_id"].tolist() == ["a", "b", "c"]
    assert result["news_type"].tolist() == ["financial_results", "industry", "financial_results"]
    assert result["prediction_score"].between(0, 1).all()
    assert model.events == [(False, False, True), (False, False, True)]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert encoded == encoded_before
    assert set(UnlabeledNewsDataset(targets["news_id"].tolist(), encoded)[0]) == {"input_ids", "attention_mask"}


@pytest.mark.parametrize("case", ["missing_id", "wrong_order", "length", "empty_ids", "float_ids", "negative_ids", "mask_shape", "nonbinary_mask", "all_padding", "padding_hole", "sequence_length"])
def test_prediction_rejects_invalid_or_misaligned_encodings_before_model_use(case):
    targets = pd.DataFrame({"news_id": ["a", "b"], "input_text": ["One", "Two"]})
    encoded = encodings(targets["news_id"])
    if case == "missing_id":
        encoded[0].pop("news_id")
    elif case == "wrong_order":
        encoded.reverse()
    elif case == "length":
        encoded.pop()
    elif case == "empty_ids":
        encoded[0]["input_ids"] = []
    elif case == "float_ids":
        encoded[0]["input_ids"] = [2.5, 1.0, 1.0]
    elif case == "negative_ids":
        encoded[0]["input_ids"] = [-1, 1, 1]
    elif case == "mask_shape":
        encoded[0]["attention_mask"] = [1]
    elif case == "nonbinary_mask":
        encoded[0]["attention_mask"] = [2, 0, 0]
    elif case == "all_padding":
        encoded[0]["attention_mask"] = [0, 0, 0]
    elif case == "padding_hole":
        encoded[0]["attention_mask"] = [1, 0, 1]
    else:
        encoded[1]["input_ids"] = [3, 1]
        encoded[1]["attention_mask"] = [1, 0]
    model = TinyInferenceModel()
    with pytest.raises(ValueError):
        predict_news_types(model, targets, encoded, "cpu", LABELS)
    assert model.events == []


@pytest.mark.parametrize("case", ["shape", "nan", "inf", "integer", "not_tensor"])
def test_prediction_rejects_invalid_model_logits(case):
    class BadModel(nn.Module):
        def forward(self, input_ids, attention_mask):
            if case == "shape":
                return torch.zeros(len(input_ids), 3)
            if case == "nan":
                return torch.full((len(input_ids), 2), float("nan"))
            if case == "inf":
                return torch.full((len(input_ids), 2), float("inf"))
            if case == "integer":
                return torch.zeros(len(input_ids), 2, dtype=torch.long)
            return [[1.0, 2.0]]
    targets = pd.DataFrame({"news_id": ["a"], "input_text": ["One"]})
    with pytest.raises(ValueError):
        predict_news_types(BadModel(), targets, encodings(["a"]), "cpu", LABELS)


def test_empty_pool_does_not_load_or_invoke_model():
    class NeverUse(nn.Module):
        def to(self, *args, **kwargs):
            raise AssertionError("Empty inference must not touch the model")
    result = predict_news_types(
        NeverUse(), pd.DataFrame(columns=["news_id", "input_text"]), [], "cpu", LABELS,
    )
    assert result.empty and result.columns.tolist() == PREDICTION_COLUMNS


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_prediction_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        predict_news_types(TinyInferenceModel(), pd.DataFrame(columns=["news_id", "input_text"]), [], "cpu", LABELS, batch_size)


@pytest.mark.parametrize("mapping", [{}, {"one": 0}, {"one": 1, "two": 2}, {"one": True, "two": 0}, {"": 0, "two": 1}])
def test_inference_rejects_invalid_label_mapping(inputs, mapping):
    inputs["label2id"] = mapping
    with pytest.raises(ValueError):
        build_inference_plan(**inputs)


def _notebook06_code_cells():
    path = Path(__file__).resolve().parents[1] / "notebooks/phase_1_news_type/06_predict_news_type.ipynb"
    return [cell for cell in json.loads(path.read_text(encoding="utf-8"))["cells"] if cell["cell_type"] == "code"]


class NotebookTinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(64, 4)

    def forward(self, input_ids, attention_mask, return_dict=True):
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class NotebookFakeTokenizer:
    pad_token_id = 1
    model_max_length = 8

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return [index % 20 + 3 for index, _ in enumerate(text.split())]

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, input_ids):
        return [0, *input_ids, 2]


@pytest.fixture
def notebook06_fixture(inputs, tmp_path, monkeypatch):
    project = tmp_path / "project"
    model_dir = project / "models/news_type_phobert_cnn/runs/20261024_153932_380173"
    reports = project / "reports/news_type/runs/20261024_153932_380173"
    (model_dir / "tokenizer").mkdir(parents=True)
    (model_dir / "backbone").mkdir()
    reports.mkdir(parents=True)
    (model_dir / "backbone/config.json").write_text("{}", encoding="utf-8")
    tables = {
        "data/processed/news_clean.csv": inputs["clean_news"],
        "data/annotation/news_type/news_manually_labeled.csv": inputs["annotations"],
        "data/annotation/news_type/label_corrections.csv": pd.DataFrame(columns=[
            "news_id", "old_news_type", "new_news_type", "rule_id", "content_sha256",
        ]),
        "data/interim/news_excluded_records.csv": inputs["excluded_records"],
        "data/interim/duplicate_merge_log.csv": inputs["duplicate_merge_log"],
    }
    split = pd.DataFrame([{
        **article("fixture-test"), "canonical_news_id": "fixture-test", "input_text": "Fixture",
        "news_type": "financial_results", "label_id": "0", "split": "test",
    }])
    for name in ("train", "validation", "test"):
        tables[f"data/splits/news_type/{name}.csv"] = split.assign(split=name)
    for name, table in tables.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, encoding="utf-8-sig")
    mapping = {"label2id": LABELS, "label_policy_version": "company_awards_listing_v2"}
    mapping_path = project / "models/news_type_phobert_cnn/label_mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    revision_path = project / "data/annotation/news_type/validation_revision.json"
    revision_path.write_text(json.dumps({"pending_reviews": [], "resolved_reviews": []}), encoding="utf-8")
    resources = {}
    for name in ("VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab", "models/wordsegmenter/wordsegmenter.rdr"):
        path = project / "models/vncorenlp" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"offline fixture resource")
        resources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    input_hashes = {
        path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [mapping_path, *[project / f"data/splits/news_type/{name}.csv" for name in ("train", "validation", "test")]]
    }
    config = {
        "model_name": "vinai/phobert-base-v2", "num_classes": 2,
        "word_segmenter": "VnCoreNLP wseg 1.2", "pooling_special_tokens": "include; padding excluded",
        "input_sha256": input_hashes, "label_mapping": mapping, "segmenter_sha256": resources,
        "max_length": 8, "head_ratio": 0.75, "kernel_sizes": [2, 3],
        "num_filters": 2, "global_projection_size": 2, "dropout": 0.1,
        "best_epoch": 1, "best_validation_macro_f1": 0.8,
    }
    (model_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    saved_model = PhoBertCNN(NotebookTinyBackbone(), 2, kernel_sizes=(2, 3), num_filters=2, global_projection_size=2, dropout=0.1)
    torch.save({"epoch": 1, "validation_macro_f1": 0.8, "model_state_dict": saved_model.state_dict()}, model_dir / "best_model.pt")
    pd.DataFrame([{
        "epoch": 1, "train_loss": 0.5, "train_accuracy": 0.8, "train_macro_f1": 0.8,
        "validation_loss": 0.5, "validation_accuracy": 0.8, "validation_macro_f1": 0.8,
    }]).to_csv(reports / "training_history.csv", index=False)
    handoff = {"source_run": model_dir.name, "checkpoint_sha256": hashlib.sha256((model_dir / "best_model.pt").read_bytes()).hexdigest(), "verified": True}
    (reports / "test_evaluation_config.json").write_text(json.dumps(handoff), encoding="utf-8")
    protected = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    fake_segmenter = ModuleType("py_vncorenlp")
    fake_segmenter.VnCoreNLP = lambda **kwargs: SimpleNamespace(word_segment=lambda text: [text])
    monkeypatch.setitem(sys.modules, "py_vncorenlp", fake_segmenter)
    events = []

    def verify_handoff(model_path, report_path, **kwargs):
        events.append("test_gate")
        path = report_path / "test_evaluation_config.json"
        if not path.is_file():
            raise FileNotFoundError("Missing completed test evaluation; finish notebook 05 first.")
        assert kwargs["label_mapping"] == mapping
        assert kwargs["test_data"]["news_id"].tolist() == ["fixture-test"]
        return json.loads(path.read_text(encoding="utf-8"))

    def encode(*args, **kwargs):
        events.append("encode")
        return encode_news_texts(*args, **kwargs)

    return SimpleNamespace(
        project=project, model_dir=model_dir, reports=reports, protected=protected, handoff=handoff,
        events=events, verify=verify_handoff, encode=encode,
    )


def _execute_notebook06(fixture, *, forbid_heavy=False):
    namespace = {
        "PROJECT_DIR": fixture.project, "IN_COLAB": False, "COLAB_OUTPUT_DIR": None,
        "pd": pd, "torch": torch, "display": lambda value: None,
        "tqdm": lambda values, **kwargs: values,
        "AutoConfig": SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(hidden_size=4)),
        "AutoModel": SimpleNamespace(from_config=lambda config: NotebookTinyBackbone()),
        "AutoTokenizer": SimpleNamespace(from_pretrained=lambda *args, **kwargs: NotebookFakeTokenizer()),
    }
    skipped = {
        "news06-colab-runtime", "news06-colab-deps", "news06-colab-drive", "news06-colab-project",
        "news06-libraries", "news06-project",
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing test handoff must stop before heavy inference or export")

    for cell in _notebook06_code_cells():
        if cell["id"] in skipped:
            continue
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        exec(source, namespace)
        if cell["id"] == "news06-imports":
            namespace["shutil"] = SimpleNamespace(which=lambda name: "offline-java")
            namespace["urlretrieve"] = forbidden
        if cell["id"] == "news06-helpers":
            namespace["verify_test_evaluation_manifest"] = fixture.verify
            namespace["encode_news_texts"] = forbidden if forbid_heavy else fixture.encode
            if forbid_heavy:
                namespace["restore_saved_checkpoint"] = forbidden
                namespace["predict_news_types"] = forbidden
    return namespace


def test_notebook06_cold_run_keeps_manual_and_hold_rows_and_exports_real_tiny_inference(notebook06_fixture):
    fixture = notebook06_fixture
    namespace = _execute_notebook06(fixture)
    output = pd.read_csv(namespace["PREDICTION_PATH"], dtype=str, keep_default_na=False)
    metadata = json.loads(namespace["PREDICTION_METADATA_PATH"].read_text(encoding="utf-8"))
    assert output.columns.tolist() == OUTPUT_COLUMNS
    assert output["news_id"].tolist() == ["manual", "target", "hold"]
    assert output["label_source"].tolist() == ["manual", "model", "review_required"]
    assert output.loc[0, "news_type"] == "financial_results"
    assert output.loc[0, "prediction_score"] == output.loc[2, "prediction_score"] == ""
    assert output.loc[2, "news_type"] == ""
    assert 0 <= float(output.loc[1, "prediction_score"]) <= 1
    assert metadata["output_rows"] == 3 and metadata["human_annotations_modified"] is False
    assert metadata["output_csv_sha256"] == hashlib.sha256(namespace["PREDICTION_PATH"].read_bytes()).hexdigest()
    assert fixture.events == ["test_gate", "encode", "test_gate"]
    for path, payload in fixture.protected.items():
        assert path.read_bytes() == payload, path


def test_notebook06_missing_test_handoff_blocks_before_encoding_weights_predictions_and_export(notebook06_fixture):
    fixture = notebook06_fixture
    (fixture.reports / "test_evaluation_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="completed test evaluation"):
        _execute_notebook06(fixture, forbid_heavy=True)
    assert fixture.events == ["test_gate"]
    assert not (fixture.project / "data/processed/news_type_predictions.csv").exists()
    assert not (fixture.project / "data/processed/news_type_predictions_metadata.json").exists()


def test_notebook06_empty_prediction_pool_exports_without_encoding_or_forward(notebook06_fixture):
    fixture = notebook06_fixture
    clean_path = fixture.project / "data/processed/news_clean.csv"
    clean = pd.read_csv(clean_path, dtype=str, keep_default_na=False).iloc[[0, 2]]
    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
    fixture.protected[clean_path] = clean_path.read_bytes()
    namespace = _execute_notebook06(fixture)
    assert fixture.events == ["test_gate", "test_gate"]
    assert namespace["prediction_inputs"].empty and namespace["model_predictions"].empty
    result = namespace["news_type_predictions"]
    assert result["label_source"].tolist() == ["manual", "review_required"]
    assert result["prediction_score"].eq("").all()
    for path, payload in fixture.protected.items():
        assert path.read_bytes() == payload, path


@pytest.mark.parametrize("preexisting_name", ["csv", "metadata"])
def test_notebook06_export_race_preserves_existing_file_and_rolls_back_owned_outputs(tmp_path, preexisting_name):
    csv_path = tmp_path / "news_type_predictions.csv"
    metadata_path = tmp_path / "news_type_predictions_metadata.json"
    protected_path = csv_path if preexisting_name == "csv" else metadata_path
    protected_path.write_bytes(b"Previously saved result; must not overwrite or delete.")
    before = protected_path.read_bytes()
    payload = b"news_id,news_type,label_source,prediction_score\n"
    namespace = {
        "PREDICTION_PATH": csv_path, "PREDICTION_METADATA_PATH": metadata_path,
        "prediction_csv_bytes": payload, "prediction_metadata_bytes": b"{}\n",
        "prediction_csv_sha256": hashlib.sha256(payload).hexdigest(), "hashlib": hashlib,
    }
    cell = next(cell for cell in _notebook06_code_cells() if cell["id"] == "news06-export")
    source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
    with pytest.raises(FileExistsError):
        exec(source, namespace)
    assert protected_path.read_bytes() == before
    other_path = metadata_path if preexisting_name == "csv" else csv_path
    assert not other_path.exists()
