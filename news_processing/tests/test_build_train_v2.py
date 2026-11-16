"""Regression checks for the hardened, hash-locked train_v2 rebuild script."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
from pathlib import Path

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "build_train_v2.py"

_spec = importlib.util.spec_from_file_location("build_train_v2", SCRIPT_PATH)
build_train_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_train_v2)

RELATIVE_PATHS = build_train_v2.RELATIVE_PATHS
EXPECTED_SHA256 = build_train_v2.EXPECTED_SHA256
TrainV2BuildError = build_train_v2.TrainV2BuildError

pytestmark = pytest.mark.skipif(
    not all((PROJECT_DIR / path).is_file() for path in RELATIVE_PATHS.values()),
    reason="Thiếu dữ liệu News Type đã khóa.",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.write_bytes(build_train_v2._csv_bytes(frame))


@pytest.fixture()
def project_copy(tmp_path: Path) -> Path:
    for relative in RELATIVE_PATHS.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_DIR / relative, target)
    return tmp_path


def _train_v2_ids(project_dir: Path) -> list[str]:
    resolution = _read(project_dir / RELATIVE_PATHS["resolution"])
    return resolution.loc[resolution["v2_action"] == "TRAIN_V2", "news_id"].tolist()


def test_real_project_reproduces_locked_train_v2_without_writing():
    output = PROJECT_DIR / RELATIVE_PATHS["output"]
    before = output.stat().st_mtime_ns

    status, train, new_rows = build_train_v2.write_train_v2(PROJECT_DIR, check_only=True)

    assert status == "unchanged"
    assert (len(train), len(new_rows)) == (740, 26)
    assert _sha256(output) == EXPECTED_SHA256["output"]
    assert output.stat().st_mtime_ns == before


def test_missing_output_is_rebuilt_byte_for_byte(project_copy: Path):
    output = project_copy / RELATIVE_PATHS["output"]
    output.unlink()

    assert build_train_v2.write_train_v2(project_copy, check_only=True)[0] == "missing"
    assert not output.exists()
    assert build_train_v2.write_train_v2(project_copy)[0] == "written"
    assert _sha256(output) == EXPECTED_SHA256["output"]
    assert not list(output.parent.glob(".*.tmp"))


def test_main_check_mode_reports_success():
    assert build_train_v2.main(["--check"]) == 0


def test_tampered_input_is_rejected_before_building(project_copy: Path):
    train_path = project_copy / RELATIVE_PATHS["train"]
    train_path.write_bytes(train_path.read_bytes() + b"\r\n")

    with pytest.raises(TrainV2BuildError, match="SHA-256 của train"):
        build_train_v2.write_train_v2(project_copy)


def test_different_existing_output_is_never_overwritten(project_copy: Path):
    output = project_copy / RELATIVE_PATHS["output"]
    frame = _read(output)
    frame.loc[0, "news_type"] = "mention_only"
    _write(frame, output)
    tampered = output.read_bytes()

    with pytest.raises(TrainV2BuildError, match="không ghi đè"):
        build_train_v2.write_train_v2(project_copy)
    assert output.read_bytes() == tampered


@pytest.mark.parametrize("key", ["news_id", "canonical_news_id", "url", "input_text"])
def test_leakage_into_validation_is_rejected(project_copy: Path, key: str):
    _, _, new_rows = build_train_v2.build_train_v2_bytes(project_copy)
    validation_path = project_copy / RELATIVE_PATHS["validation"]
    validation = _read(validation_path)
    leaked = new_rows.iloc[0][key]
    validation.loc[0, key] = f"  {leaked}  " if key == "input_text" else leaked
    _write(validation, validation_path)

    with pytest.raises(TrainV2BuildError, match=f"validation theo {key}"):
        build_train_v2.build_train_v2_bytes(project_copy, expected_sha256=None)


def test_content_drift_after_review_is_rejected(project_copy: Path):
    clean_path = project_copy / RELATIVE_PATHS["clean"]
    clean = _read(clean_path)
    target = _train_v2_ids(project_copy)[0]
    clean.loc[clean["news_id"] == target, "content"] += " Thêm câu."
    _write(clean, clean_path)

    with pytest.raises(TrainV2BuildError, match="reviewed_content_sha256"):
        build_train_v2.build_train_v2_bytes(project_copy, expected_sha256=None)


def test_unlabeled_train_row_is_rejected(project_copy: Path):
    resolution_path = project_copy / RELATIVE_PATHS["resolution"]
    resolution = _read(resolution_path)
    target = _train_v2_ids(project_copy)[0]
    resolution.loc[resolution["news_id"] == target, "final_annotation_status"] = "review_required"
    _write(resolution, resolution_path)

    with pytest.raises(TrainV2BuildError, match="labeled"):
        build_train_v2.build_train_v2_bytes(project_copy, expected_sha256=None)


def test_changed_disposition_counts_are_rejected(project_copy: Path):
    resolution_path = project_copy / RELATIVE_PATHS["resolution"]
    resolution = _read(resolution_path)
    target = _train_v2_ids(project_copy)[0]
    resolution.loc[resolution["news_id"] == target, "v2_action"] = "HELDOUT_OVERLAP"
    _write(resolution, resolution_path)

    with pytest.raises(TrainV2BuildError, match="v2_action"):
        build_train_v2.build_train_v2_bytes(project_copy, expected_sha256=None)
