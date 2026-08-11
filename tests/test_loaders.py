"""Unit tests for src/data/loaders.py.

Uses small synthetic/fixture CSVs written to tmp_path and monkeypatched onto
the module's path constants, so this suite never requires the real (large,
gitignored) files under data/raw/ — see docs/data_contract.md section 2 for
why those are not committed. This keeps the tests runnable in CI.
"""
import pandas as pd
import pytest

from src.data import loaders
from src.data.schemas import (
    DEMOGRAPHICS_COLUMNS,
    HEALTH_PROVIDER_LABEL_COL,
    MOTOR_ALL_COLUMNS,
    MOTOR_TARGET,
)


def _write_csv(path, df: pd.DataFrame):
    df.to_csv(path, index=False)
    return path


def test_load_motor_claims_returns_dataframe_with_target(tmp_path, monkeypatch):
    row = {c: 0 for c in MOTOR_ALL_COLUMNS}
    df = pd.DataFrame([row, {**row, MOTOR_TARGET: 1}])
    path = _write_csv(tmp_path / "fraud_oracle.csv", df)
    monkeypatch.setattr(loaders, "MOTOR_RAW_PATH", path)

    loaded = loaders.load_motor_claims()
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape[0] > 0
    assert MOTOR_TARGET in loaded.columns


def test_load_demographics_reference_returns_expected_columns(tmp_path, monkeypatch):
    df = pd.DataFrame([{c: 0 for c in DEMOGRAPHICS_COLUMNS}])
    path = _write_csv(tmp_path / "Indian_Insurance_Data.csv", df)
    monkeypatch.setattr(loaders, "DEMOGRAPHICS_RAW_PATH", path)

    loaded = loaders.load_demographics_reference()
    assert list(loaded.columns) == DEMOGRAPHICS_COLUMNS


def test_load_health_train_tables_has_label_on_main_only(tmp_path, monkeypatch):
    main = pd.DataFrame({"Provider": ["P1", "P2"], HEALTH_PROVIDER_LABEL_COL: ["Yes", "No"]})
    beneficiary = pd.DataFrame({"BeneID": ["B1"]})
    inpatient = pd.DataFrame({"ClaimID": ["CLM4001"]})
    outpatient = pd.DataFrame({"ClaimID": ["CLM6001"]})

    monkeypatch.setattr(
        loaders, "HEALTH_TRAIN_MAIN_PATH", _write_csv(tmp_path / "train_main.csv", main)
    )
    monkeypatch.setattr(
        loaders,
        "HEALTH_TRAIN_BENEFICIARY_PATH",
        _write_csv(tmp_path / "train_bene.csv", beneficiary),
    )
    monkeypatch.setattr(
        loaders,
        "HEALTH_TRAIN_INPATIENT_PATH",
        _write_csv(tmp_path / "train_in.csv", inpatient),
    )
    monkeypatch.setattr(
        loaders,
        "HEALTH_TRAIN_OUTPATIENT_PATH",
        _write_csv(tmp_path / "train_out.csv", outpatient),
    )

    tables = loaders.load_health_train_tables()
    assert HEALTH_PROVIDER_LABEL_COL in tables.main.columns
    assert tables.beneficiary.shape[0] > 0
    assert tables.inpatient.shape[0] > 0
    assert tables.outpatient.shape[0] > 0


def test_load_health_test_main_has_no_label(tmp_path, monkeypatch):
    main = pd.DataFrame({"Provider": ["P1", "P2"]})
    beneficiary = pd.DataFrame({"BeneID": ["B1"]})
    inpatient = pd.DataFrame({"ClaimID": ["CLM4001"]})
    outpatient = pd.DataFrame({"ClaimID": ["CLM6001"]})

    monkeypatch.setattr(
        loaders, "HEALTH_TEST_MAIN_PATH", _write_csv(tmp_path / "test_main.csv", main)
    )
    monkeypatch.setattr(
        loaders,
        "HEALTH_TEST_BENEFICIARY_PATH",
        _write_csv(tmp_path / "test_bene.csv", beneficiary),
    )
    monkeypatch.setattr(
        loaders, "HEALTH_TEST_INPATIENT_PATH", _write_csv(tmp_path / "test_in.csv", inpatient)
    )
    monkeypatch.setattr(
        loaders,
        "HEALTH_TEST_OUTPATIENT_PATH",
        _write_csv(tmp_path / "test_out.csv", outpatient),
    )

    tables = loaders.load_health_test_tables()
    assert HEALTH_PROVIDER_LABEL_COL not in tables.main.columns


def test_read_csv_missing_file_raises_actionable_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(loaders.DatasetFileMissingError, match="not found"):
        loaders._read_csv(missing_path)
