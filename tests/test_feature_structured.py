import numpy as np
import pandas as pd

from src.data.schemas import MOTOR_ALL_COLUMNS, MOTOR_TARGET
from src.features import structured


def _tiny_motor_df():
    row = {c: 0 for c in MOTOR_ALL_COLUMNS}
    row["Make"] = "Honda"
    return pd.DataFrame(
        [row, {**row, MOTOR_TARGET: 1}, {**row, MOTOR_TARGET: 0}],
        index=[10, 11, 12],
    )


def _tiny_health_tables():
    inpatient = pd.DataFrame(
        {
            "ClaimID": ["CLM100"],
            "BeneID": ["BENE1"],
            "ClmDiagnosisCode_1": ["D1"],
            "ClmDiagnosisCode_2": [np.nan],
        }
    )
    outpatient = pd.DataFrame(
        {
            "ClaimID": ["CLM200", "CLM201"],
            "BeneID": ["BENE1", "BENE2"],
            "ClmDiagnosisCode_1": [np.nan, "D9"],
            "ClmDiagnosisCode_2": [np.nan, np.nan],
        }
    )
    beneficiary = pd.DataFrame(
        {
            "BeneID": ["BENE1", "BENE2"],
            "DOD": [None, "2009-01-01"],
            "ChronicCond_Alzheimer": [1, 2],
            "RenalDiseaseIndicator": ["0", "Y"],
        }
    )
    return inpatient, outpatient, beneficiary


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------


def test_build_motor_structured_features_claim_key_matches_row_index():
    out = structured.build_motor_structured_features(_tiny_motor_df())
    assert list(out[structured.CLAIM_KEY_COL]) == ["claim:10", "claim:11", "claim:12"]


def test_build_motor_structured_features_excludes_target_and_ids():
    out = structured.build_motor_structured_features(_tiny_motor_df())
    assert MOTOR_TARGET not in out.columns
    assert "PolicyNumber" not in out.columns
    assert "RepNumber" not in out.columns


def test_build_motor_structured_features_preserves_categorical_dtype():
    out = structured.build_motor_structured_features(_tiny_motor_df())
    assert str(out["Make"].dtype) == "category"


def test_build_motor_structured_features_does_not_mutate_input():
    df = _tiny_motor_df()
    before = df.copy(deep=True)
    structured.build_motor_structured_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_build_motor_structured_features_deterministic():
    df = _tiny_motor_df()
    out1 = structured.build_motor_structured_features(df)
    out2 = structured.build_motor_structured_features(df)
    pd.testing.assert_frame_equal(out1, out2)


def test_build_motor_structured_features_drops_leakage_columns_defensively():
    df = _tiny_motor_df()
    df["synthetic_fraud_ring_label"] = 1
    df["synthetic_ring_id"] = "R1"
    out = structured.build_motor_structured_features(df)
    assert "synthetic_fraud_ring_label" not in out.columns
    assert "synthetic_ring_id" not in out.columns


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_build_health_structured_features_claim_key_format():
    inpatient, outpatient, beneficiary = _tiny_health_tables()
    out = structured.build_health_structured_features(inpatient, outpatient, beneficiary)
    assert set(out[structured.CLAIM_KEY_COL]) == {"claim:CLM100", "claim:CLM200", "claim:CLM201"}


def test_build_health_structured_features_merges_beneficiary_derived_columns():
    inpatient, outpatient, beneficiary = _tiny_health_tables()
    out = structured.build_health_structured_features(inpatient, outpatient, beneficiary)
    row = out.loc[out[structured.CLAIM_KEY_COL] == "claim:CLM201"].iloc[0]
    assert row["is_deceased"] == True  # noqa: E712 (BENE2 has a DOD)
    assert row["has_renal_disease"] == True  # noqa: E712 (BENE2 is 'Y')
    assert row["chronic_condition_count"] == 0  # BENE2's ChronicCond_Alzheimer == 2 (absent)


def test_build_health_structured_features_adds_claim_type_and_code_counts():
    inpatient, outpatient, beneficiary = _tiny_health_tables()
    out = structured.build_health_structured_features(inpatient, outpatient, beneficiary)
    assert set(out["claim_type"]) == {"inpatient", "outpatient"}
    row = out.loc[out[structured.CLAIM_KEY_COL] == "claim:CLM100"].iloc[0]
    assert row["num_diagnosis_codes"] == 1


def test_build_health_structured_features_excludes_identifiers_and_labels():
    inpatient, outpatient, beneficiary = _tiny_health_tables()
    out = structured.build_health_structured_features(inpatient, outpatient, beneficiary)
    assert "ClaimID" not in out.columns
    assert "BeneID" not in out.columns
    assert "claim_fraud_label_provider_derived" not in out.columns


def test_build_health_structured_features_drops_leakage_column_if_preattached():
    inpatient, outpatient, beneficiary = _tiny_health_tables()
    inpatient = inpatient.copy()
    inpatient["claim_fraud_label_provider_derived"] = 1
    out = structured.build_health_structured_features(inpatient, outpatient, beneficiary)
    assert "claim_fraud_label_provider_derived" not in out.columns


def test_build_health_structured_features_does_not_mutate_inputs():
    inpatient, outpatient, beneficiary = _tiny_health_tables()
    ip_before, op_before, bene_before = inpatient.copy(deep=True), outpatient.copy(deep=True), beneficiary.copy(deep=True)
    structured.build_health_structured_features(inpatient, outpatient, beneficiary)
    pd.testing.assert_frame_equal(inpatient, ip_before)
    pd.testing.assert_frame_equal(outpatient, op_before)
    pd.testing.assert_frame_equal(beneficiary, bene_before)


def test_build_health_structured_features_deterministic():
    inpatient, outpatient, beneficiary = _tiny_health_tables()
    out1 = structured.build_health_structured_features(inpatient, outpatient, beneficiary)
    out2 = structured.build_health_structured_features(inpatient, outpatient, beneficiary)
    pd.testing.assert_frame_equal(out1, out2)


def test_build_health_structured_features_duplicate_beneficiary_rows_do_not_fan_out():
    inpatient, outpatient, beneficiary = _tiny_health_tables()
    dup_beneficiary = pd.concat([beneficiary, beneficiary.iloc[[0]]], ignore_index=True)
    out = structured.build_health_structured_features(inpatient, outpatient, dup_beneficiary)
    assert len(out) == 3  # still one row per claim, not fanned out by the duplicate
