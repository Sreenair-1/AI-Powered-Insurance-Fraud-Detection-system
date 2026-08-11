import pandas as pd
import pytest

from src.data import loaders
from src.data.schemas import MOTOR_TARGET, HEALTH_PROVIDER_LABEL_COL, DEMOGRAPHICS_COLUMNS


def test_load_motor_claims_returns_dataframe_with_target():
    df = loaders.load_motor_claims()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert MOTOR_TARGET in df.columns


def test_load_demographics_reference_returns_expected_columns():
    df = loaders.load_demographics_reference()
    assert list(df.columns) == DEMOGRAPHICS_COLUMNS


def test_load_health_train_tables_has_label_on_main_only():
    tables = loaders.load_health_train_tables()
    assert HEALTH_PROVIDER_LABEL_COL in tables.main.columns
    assert tables.beneficiary.shape[0] > 0
    assert tables.inpatient.shape[0] > 0
    assert tables.outpatient.shape[0] > 0


def test_load_health_test_main_has_no_label():
    tables = loaders.load_health_test_tables()
    assert HEALTH_PROVIDER_LABEL_COL not in tables.main.columns


def test_read_csv_missing_file_raises_actionable_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(loaders.DatasetFileMissingError, match="not found"):
        loaders._read_csv(missing_path)
