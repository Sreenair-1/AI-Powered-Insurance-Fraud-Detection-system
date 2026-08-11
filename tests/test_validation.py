import pandas as pd

from src.data import validation
from src.data.schemas import MOTOR_TARGET


def _tiny_motor_df():
    from src.data.schemas import MOTOR_ALL_COLUMNS

    row = {c: 0 for c in MOTOR_ALL_COLUMNS}
    row[MOTOR_TARGET] = 0
    return pd.DataFrame([row, {**row, MOTOR_TARGET: 1}])


def test_validate_motor_valid_dataframe_passes():
    result = validation.validate_motor(_tiny_motor_df())
    assert result.ok
    assert result.errors == []


def test_validate_motor_missing_column_fails():
    df = _tiny_motor_df().drop(columns=[MOTOR_TARGET])
    result = validation.validate_motor(df)
    assert not result.ok
    assert any("missing expected columns" in e for e in result.errors)


def test_validate_motor_empty_dataframe_fails():
    df = _tiny_motor_df().iloc[0:0]
    result = validation.validate_motor(df)
    assert not result.ok
    assert any("empty" in e for e in result.errors)


def test_validate_health_main_valid_labels_pass():
    df = pd.DataFrame({"Provider": ["P1", "P2"], "PotentialFraud": ["Yes", "No"]})
    result = validation.validate_health_main(df)
    assert result.ok
    assert result.stats["provider_positive_rate"] == 0.5


def test_validate_health_main_unexpected_label_value_fails():
    df = pd.DataFrame({"Provider": ["P1"], "PotentialFraud": ["Maybe"]})
    result = validation.validate_health_main(df)
    assert not result.ok


def test_validate_health_main_missing_label_warns_not_fails():
    df = pd.DataFrame({"Provider": ["P1", "P2"]})
    result = validation.validate_health_main(df)
    assert result.ok
    assert any("not usable for supervised" in w for w in result.warnings)


def test_count_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    assert validation.count_duplicate_rows(df) == 1


def test_validate_health_claims_detects_duplicate_claim_ids():
    df = pd.DataFrame(
        {
            "BeneID": ["B1", "B2"],
            "ClaimID": ["C1", "C1"],
            "Provider": ["P1", "P1"],
        }
    )
    result = validation.validate_health_claims(df)
    assert not result.ok
    assert any("duplicate ClaimID" in e for e in result.errors)
