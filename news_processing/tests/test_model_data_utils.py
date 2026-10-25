import json
from pathlib import Path

import pandas as pd
import pytest

from src.model_data_utils import (
    MODEL_SPLIT_COLUMNS,
    build_phobert_input_text,
    build_split_distribution_report,
    compute_balanced_class_weights,
    is_redundant_summary,
    prepare_eligible_model_dataset,
    reuse_split_membership,
    stratified_train_val_test_split,
    verify_model_splits,
)


LABEL2ID = {
    "governance_capital": 0,
    "financial_results": 1,
    "business_operations": 2,
    "legal_compliance": 3,
    "industry": 4,
    "stock_market": 5,
    "mention_only": 6,
}


def test_build_phobert_input_text_deduplicates_repeated_opening_summary():
    title = "Doanh thu quý I của Vinamilk tăng 24,6%"
    summary = "Báo cáo kết quả kinh doanh quý 1 năm nay của Vinamilk cho thấy doanh thu thuần hợp nhất đạt 16.149 tỷ đồng."
    content = (
        f"{summary} Kết quả kinh doanh tăng trưởng tích cực nhờ thị trường nội địa hồi phục."
    )

    assert is_redundant_summary(title, summary, content)
    input_text = build_phobert_input_text(title, summary, content)
    assert input_text.startswith("Doanh thu quý I của Vinamilk tăng 24,6%.")
    assert input_text.count("16.149 tỷ đồng") == 1


def test_build_phobert_input_text_keeps_distinct_summary():
    title = "Nhóm quỹ Dragon Capital giảm sở hữu tại Thế Giới Di Động"
    summary = "Cùng chiều, ông Đoàn Văn Hiểu Em cũng đăng ký bán 2 triệu cổ phiếu MWG."
    content = "Nhóm quỹ Dragon Capital vừa thông báo hoàn tất bán ròng 541.366 cổ phiếu MWG."

    assert not is_redundant_summary(title, summary, content)
    input_text = build_phobert_input_text(title, summary, content)
    assert summary in input_text
    assert content in input_text


def test_stratified_split_and_leakage_verification():
    rows = []
    labels = list(LABEL2ID.keys()) + ["macro"]
    for i in range(160):
        raw_label = labels[i % len(labels)]
        rows.append({
            "news_id": f"id-{i:04d}",
            "source": "CafeF",
            "url": f"https://example.com/article-{i}",
            "title": f"Tiêu đề bài viết độc lập số {i}",
            "summary": f"Tóm tắt riêng biệt số {i}",
            "content": f"Nội dung chi tiết của bài báo kinh tế số {i} dài hơn 200 ký tự.",
            "published_date": "2025-05-10",
            "tickers": "MWG",
            "news_type": raw_label,
            "annotation_status": "labeled",
            "annotator_note": "",
        })
    annotations = pd.DataFrame(rows)
    clean_news = annotations[
        ["news_id", "source", "url", "title", "summary", "content", "published_date", "tickers"]
    ].copy()
    excluded = pd.DataFrame(columns=["news_id", "content", "exclusion_reason"])
    merges = pd.DataFrame(columns=["kept_news_id", "merged_news_ids"])

    eligible, _ = prepare_eligible_model_dataset(
        annotations, clean_news, excluded, merges, LABEL2ID
    )
    assert len(eligible) == 160
    assert set(eligible["news_type"]) == set(LABEL2ID.keys())

    train_df, val_df, test_df = stratified_train_val_test_split(
        eligible, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )
    assert len(train_df) + len(val_df) + len(test_df) == 160
    checks = verify_model_splits(eligible, train_df, val_df, test_df, LABEL2ID)
    assert checks["issue_count"].eq(0).all()

    dist = build_split_distribution_report(train_df, val_df, test_df, LABEL2ID)
    assert len(dist) == 7
    assert int(dist["total_count"].sum()) == 160

    weights = compute_balanced_class_weights(train_df, LABEL2ID)
    assert len(weights) == 7
    assert weights["class_weight"].gt(0).all()


@pytest.fixture
def fixed_model_splits():
    rows = []
    labels = list(LABEL2ID)
    for index in range(21):
        label = labels[index % len(labels)]
        rows.append({
            "news_id": f"fixed-{index}",
            "canonical_news_id": f"fixed-{index}",
            "source": "CafeF",
            "url": f"https://example.com/fixed-{index}",
            "published_date": "2025-01-01",
            "tickers": "PNJ",
            "title": f"Distinct article {index}",
            "summary": f"Summary {index}",
            "content": f"Independent content {index}",
            "input_text": f"Original model input {index}",
            "news_type": label,
            "label_id": LABEL2ID[label],
        })
    data = pd.DataFrame(rows)
    splits = {}
    for offset, split_name in enumerate(("train", "validation", "test")):
        frame = data.iloc[offset * 7:(offset + 1) * 7].iloc[::-1].copy()
        frame["split"] = split_name
        splits[split_name] = frame.reset_index(drop=True)
    return data, splits


def test_reuse_split_membership_refreshes_rows_without_reordering(fixed_model_splits):
    data, saved_splits = fixed_model_splits
    original_data = data.copy(deep=True)
    original_splits = {name: frame.copy(deep=True) for name, frame in saved_splits.items()}
    updated = data.sample(frac=1, random_state=77).copy()
    corrected = updated["news_id"].eq("fixed-0")
    updated.loc[corrected, ["news_type", "label_id"]] = ["industry", 4]
    updated.loc[corrected, "input_text"] = "Refreshed model input"
    updated["summary_redundant"] = False

    rebuilt = reuse_split_membership(updated, saved_splits)

    for (split_name, previous), current in zip(saved_splits.items(), rebuilt):
        assert current["news_id"].tolist() == previous["news_id"].tolist()
        assert current["split"].eq(split_name).all()
        assert current.columns.tolist() == MODEL_SPLIT_COLUMNS + ["split"]
        assert current.index.tolist() == list(range(len(current)))
    corrected_row = rebuilt[0].set_index("news_id").loc["fixed-0"]
    assert corrected_row["news_type"] == "industry"
    assert corrected_row["label_id"] == 4
    assert corrected_row["input_text"] == "Refreshed model input"
    pd.testing.assert_frame_equal(data, original_data)
    for name in saved_splits:
        pd.testing.assert_frame_equal(saved_splits[name], original_splits[name])

    repeated = reuse_split_membership(
        updated, dict(zip(("train", "validation", "test"), rebuilt))
    )
    for first, second in zip(rebuilt, repeated):
        pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("keys", [(), ("train",), ("train", "validation"),
                                  ("train", "validation", "test", "extra")])
def test_reuse_split_membership_requires_all_three_splits(fixed_model_splits, keys):
    data, splits = fixed_model_splits
    selected = {name: splits.get(name, splits["train"]) for name in keys}
    with pytest.raises(ValueError, match="exactly train, validation, and test"):
        reuse_split_membership(data, selected)


@pytest.mark.parametrize("invalid_id", [None, "", "   "])
def test_reuse_split_membership_rejects_empty_current_ids(fixed_model_splits, invalid_id):
    data, splits = fixed_model_splits
    data.loc[0, "news_id"] = invalid_id
    with pytest.raises(ValueError, match="nonempty and unique in current data"):
        reuse_split_membership(data, splits)


def test_reuse_split_membership_rejects_duplicate_current_ids(fixed_model_splits):
    data, splits = fixed_model_splits
    data.loc[1, "news_id"] = data.loc[0, "news_id"]
    with pytest.raises(ValueError, match="nonempty and unique in current data"):
        reuse_split_membership(data, splits)


@pytest.mark.parametrize("problem", ["empty", "duplicate", "missing_id", "empty_frame"])
def test_reuse_split_membership_rejects_invalid_saved_ids(fixed_model_splits, problem):
    data, splits = fixed_model_splits
    if problem == "empty":
        splits["train"].loc[0, "news_id"] = " "
    elif problem == "duplicate":
        splits["train"].loc[1, "news_id"] = splits["train"].loc[0, "news_id"]
    elif problem == "missing_id":
        splits["train"] = splits["train"].drop(columns="news_id")
    else:
        splits["train"] = splits["train"].iloc[:0]
    with pytest.raises(ValueError, match="news_id"):
        reuse_split_membership(data, splits)


@pytest.mark.parametrize("problem", ["missing", "wrong", "null"])
def test_reuse_split_membership_rejects_invalid_split_tags(fixed_model_splits, problem):
    data, splits = fixed_model_splits
    if problem == "missing":
        splits["validation"] = splits["validation"].drop(columns="split")
    else:
        splits["validation"].loc[0, "split"] = "train" if problem == "wrong" else None
    with pytest.raises(ValueError, match="Invalid split tags"):
        reuse_split_membership(data, splits)


def test_reuse_split_membership_rejects_overlap(fixed_model_splits):
    data, splits = fixed_model_splits
    splits["validation"].loc[0, "news_id"] = splits["train"].loc[0, "news_id"]
    with pytest.raises(ValueError, match="memberships overlap"):
        reuse_split_membership(data, splits)


@pytest.mark.parametrize("problem", ["extra_saved", "missing_saved", "new_eligible"])
def test_reuse_split_membership_rejects_eligibility_drift(fixed_model_splits, problem):
    data, splits = fixed_model_splits
    if problem == "extra_saved":
        splits["test"].loc[0, "news_id"] = "unknown-article"
    elif problem == "missing_saved":
        splits["test"] = splits["test"].iloc[1:]
    else:
        new_row = data.iloc[[0]].copy()
        new_row["news_id"] = "newly-eligible"
        data = pd.concat([data, new_row], ignore_index=True)
    with pytest.raises(ValueError, match="exactly cover current eligible data"):
        reuse_split_membership(data, splits)


def test_reuse_split_membership_rejects_missing_current_columns(fixed_model_splits):
    data, splits = fixed_model_splits
    with pytest.raises(ValueError, match="missing model columns"):
        reuse_split_membership(data.drop(columns="input_text"), splits)


def test_verify_model_splits_accepts_valid_fixed_membership(fixed_model_splits):
    data, splits = fixed_model_splits
    checks = verify_model_splits(data, *splits.values(), LABEL2ID)
    assert checks["issue_count"].eq(0).all()


def test_verify_model_splits_detects_duplicate_ids_and_missing_coverage(fixed_model_splits):
    data, splits = fixed_model_splits
    splits["train"].iloc[1] = splits["train"].iloc[0]
    checks = verify_model_splits(data, *splits.values(), LABEL2ID).set_index("check")
    assert checks.at["news_id trùng trong cùng một tập", "issue_count"] == 1
    assert checks.at["news_id thiếu hoặc ngoài dữ liệu đủ điều kiện", "issue_count"] == 1


def test_verify_model_splits_detects_substituted_id_without_count_change(fixed_model_splits):
    data, splits = fixed_model_splits
    splits["train"].loc[0, "news_id"] = "unknown-article"
    checks = verify_model_splits(data, *splits.values(), LABEL2ID).set_index("check")
    assert checks.at["news_id thiếu hoặc ngoài dữ liệu đủ điều kiện", "issue_count"] == 2


@pytest.mark.parametrize("invalid_label", ["wrong_id", "unknown_type", "invalid_id"])
def test_verify_model_splits_detects_label_mapping_mismatch(fixed_model_splits, invalid_label):
    data, splits = fixed_model_splits
    train = splits["train"]
    if invalid_label == "wrong_id":
        train.loc[0, "label_id"] = (train.loc[0, "label_id"] + 1) % len(LABEL2ID)
    elif invalid_label == "unknown_type":
        train.loc[0, "news_type"] = "not_a_label"
    else:
        train["label_id"] = train["label_id"].astype(object)
        train.loc[0, "label_id"] = "not_an_integer"
    checks = verify_model_splits(data, *splits.values(), LABEL2ID).set_index("check")
    assert checks.at["news_type không khớp label_id", "issue_count"] == 1


@pytest.mark.parametrize("problem", ["duplicate", "empty"])
def test_verify_model_splits_detects_invalid_eligible_ids(fixed_model_splits, problem):
    data, splits = fixed_model_splits
    data.loc[0, "news_id"] = data.loc[1, "news_id"] if problem == "duplicate" else " "
    checks = verify_model_splits(data, *splits.values(), LABEL2ID).set_index("check")
    assert checks.at["news_id rỗng hoặc trùng trong dữ liệu đủ điều kiện", "issue_count"] == 1


def _prepare_model_data_notebook():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks/phase_1_news_type/04_prepare_model_data.ipynb"
    )
    return json.loads(notebook_path.read_text(encoding="utf-8"))


def _prepare_model_data_cell(cell_id):
    return next(
        "".join(cell["source"])
        for cell in _prepare_model_data_notebook()["cells"]
        if cell["id"] == cell_id
    )


def test_notebook_reuses_saved_membership_without_random_splitting(fixed_model_splits):
    data, saved_splits = fixed_model_splits
    data.loc[data["news_id"].eq("fixed-0"), ["news_type", "label_id"]] = ["industry", 4]

    def forbid_new_split(*args, **kwargs):
        raise AssertionError("Existing split membership must not be randomized again")

    namespace = {
        "eligible_news": data,
        "MODEL_SPLIT_COLUMNS": MODEL_SPLIT_COLUMNS,
        "existing_splits": saved_splits,
        "reuse_split_membership": reuse_split_membership,
        "stratified_train_val_test_split": forbid_new_split,
    }
    exec(_prepare_model_data_cell("clear-13"), namespace)

    result_names = ("train_data", "validation_data", "test_data")
    for (split_name, saved), result_name in zip(saved_splits.items(), result_names):
        rebuilt = namespace[result_name]
        assert rebuilt["news_id"].tolist() == saved["news_id"].tolist()
        assert rebuilt["split"].eq(split_name).all()
    corrected = namespace["train_data"].set_index("news_id").loc["fixed-0"]
    assert corrected["news_type"] == "industry"
    assert corrected["label_id"] == 4


def test_notebook_only_creates_new_splits_when_none_exist(fixed_model_splits):
    data, saved_splits = fixed_model_splits
    split_arguments = {}

    def capture_new_split(current_data, **kwargs):
        pd.testing.assert_frame_equal(current_data, data[MODEL_SPLIT_COLUMNS])
        split_arguments.update(kwargs)
        return tuple(saved_splits.values())

    def forbid_reuse(*args, **kwargs):
        raise AssertionError("No saved memberships are available")

    namespace = {
        "eligible_news": data,
        "MODEL_SPLIT_COLUMNS": MODEL_SPLIT_COLUMNS,
        "existing_splits": {},
        "reuse_split_membership": forbid_reuse,
        "stratified_train_val_test_split": capture_new_split,
        "TRAIN_RATIO": 0.60,
        "VALIDATION_RATIO": 0.20,
        "TEST_RATIO": 0.20,
        "RANDOM_SEED": 103,
    }
    exec(_prepare_model_data_cell("clear-13"), namespace)
    assert split_arguments == {
        "train_ratio": 0.60, "val_ratio": 0.20, "test_ratio": 0.20,
        "group_column": "label_id", "random_seed": 103,
    }
    assert namespace["train_data"] is saved_splits["train"]
    assert namespace["validation_data"] is saved_splits["validation"]
    assert namespace["test_data"] is saved_splits["test"]


@pytest.mark.parametrize("file_count", [0, 1, 2, 3])
def test_notebook_rejects_partial_saved_split_sets(monkeypatch, fixed_model_splits, file_count):
    _, saved_splits = fixed_model_splits
    source = _prepare_model_data_cell("clear-existing-splits")
    split_directory = Path("synthetic-split-directory")
    available_paths = {
        split_directory / f"{name}.csv" for name in tuple(saved_splits)[:file_count]
    }
    read_paths = []

    def fake_read_csv(path, **kwargs):
        assert kwargs == {"dtype": str, "keep_default_na": False}
        read_paths.append(path)
        return saved_splits[path.stem].copy()

    monkeypatch.setattr(Path, "is_file", lambda path: path in available_paths)
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    namespace = {"SPLIT_DIR": split_directory, "pd": pd}
    if file_count in (1, 2):
        with pytest.raises(FileNotFoundError, match="không tự chia lại dữ liệu"):
            exec(source, namespace)
        assert not read_paths
    else:
        exec(source, namespace)
        assert len(namespace["existing_splits"]) == file_count
        assert set(read_paths) == available_paths


def test_prepare_model_data_notebook_is_valid_and_cells_are_focused():
    import nbformat

    notebook = _prepare_model_data_notebook()
    nbformat.validate(nbformat.from_dict(notebook))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        assert len(source.splitlines()) <= 21, cell["id"]
        compile(source, f"notebook04:{cell['id']}", "exec")

