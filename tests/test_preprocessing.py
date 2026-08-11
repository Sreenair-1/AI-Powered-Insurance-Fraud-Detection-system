import numpy as np
import pandas as pd

from src.data import preprocessing
from src.data.schemas import (
    HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL,
    MOTOR_ID_COLUMNS,
    MOTOR_TARGET,
)


def _tiny_motor_df():
    from src.data.schemas import MOTOR_ALL_COLUMNS

    row = {c: 0 for c in MOTOR_ALL_COLUMNS}
    row["Make"] = "Honda"
    return pd.DataFrame([row, {**row, MOTOR_TARGET: 1}])


def test_build_motor_feature_target_drops_ids_and_target():
    X, y = preprocessing.build_motor_feature_target(_tiny_motor_df())
    assert MOTOR_TARGET not in X.columns
    for id_col in MOTOR_ID_COLUMNS:
        assert id_col not in X.columns
    assert list(y) == [0, 1]


def test_build_motor_feature_target_encodes_categoricals():
    X, _ = preprocessing.build_motor_feature_target(_tiny_motor_df())
    assert str(X["Make"].dtype) == "category"


def test_add_beneficiary_derived_features():
    df = pd.DataFrame(
        {
            "DOD": [None, "2009-01-01"],
            "ChronicCond_Alzheimer": [1, 2],
            "ChronicCond_Heartfailure": [1, 1],
            "RenalDiseaseIndicator": ["0", "Y"],
        }
    )
    out = preprocessing.add_beneficiary_derived_features(df)
    assert list(out["is_deceased"]) == [False, True]
    assert list(out["chronic_condition_count"]) == [2, 1]
    assert list(out["has_renal_disease"]) == [False, True]


def test_build_health_claims_table_adds_claim_type_and_code_counts():
    inpatient = pd.DataFrame(
        {"ClaimID": ["C1"], "ClmDiagnosisCode_1": ["D1"], "ClmDiagnosisCode_2": [np.nan]}
    )
    outpatient = pd.DataFrame(
        {"ClaimID": ["C2"], "ClmDiagnosisCode_1": [np.nan], "ClmDiagnosisCode_2": [np.nan]}
    )
    combined = preprocessing.build_health_claims_table(inpatient, outpatient)
    assert set(combined["claim_type"]) == {"inpatient", "outpatient"}
    assert combined.loc[combined["ClaimID"] == "C1", "num_diagnosis_codes"].item() == 1
    assert combined.loc[combined["ClaimID"] == "C2", "num_diagnosis_codes"].item() == 0


def test_build_claim_fraud_label_provider_derived_maps_yes_no():
    claims = pd.DataFrame({"ClaimID": ["C1", "C2"], "Provider": ["P1", "P2"]})
    provider_labels = pd.DataFrame({"Provider": ["P1", "P2"], "PotentialFraud": ["Yes", "No"]})
    out = preprocessing.build_claim_fraud_label_provider_derived(claims, provider_labels)
    assert HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL in out.columns
    assert "PotentialFraud" not in out.columns
    assert list(out[HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL]) == [1, 0]
