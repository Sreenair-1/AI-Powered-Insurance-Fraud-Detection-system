"""Deterministic feature preparation. No step here may be fit on validation/test data.

Every function is a pure transform of its input dataframe(s) — callers are
responsible for calling health provider-aggregate style functions per-split
(after splitting), never on the full unsplit data, to avoid leakage
(see docs/data_contract.md section 9).
"""
from __future__ import annotations

import pandas as pd

from src.data.schemas import (
    HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL,
    HEALTH_BENEFICIARY_CHRONIC_COLUMNS,
    HEALTH_DIAGNOSIS_CODE_COLUMNS,
    HEALTH_PROCEDURE_CODE_COLUMNS,
    HEALTH_PROVIDER_COL,
    HEALTH_PROVIDER_LABEL_COL,
    MOTOR_CATEGORICAL_COLUMNS,
    MOTOR_ID_COLUMNS,
    MOTOR_TARGET,
)

# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------


def encode_motor_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known categorical columns to pandas 'category' dtype (no value-derived fitting)."""
    out = df.copy()
    for col in MOTOR_CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def build_motor_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split motor claims into (X, y), dropping identifier columns and the target from X."""
    y = df[MOTOR_TARGET].astype(int)
    X = df.drop(columns=[MOTOR_TARGET, *MOTOR_ID_COLUMNS], errors="ignore")
    return encode_motor_categoricals(X), y


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def add_beneficiary_derived_features(beneficiary_df: pd.DataFrame) -> pd.DataFrame:
    """Add is_deceased and chronic_condition_count; leaves raw columns untouched."""
    out = beneficiary_df.copy()
    out["is_deceased"] = out["DOD"].notna()
    present_chronic_cols = [c for c in HEALTH_BENEFICIARY_CHRONIC_COLUMNS if c in out.columns]
    # source encoding: 1 = condition present, 2 = absent
    out["chronic_condition_count"] = (out[present_chronic_cols] == 1).sum(axis=1)
    if "RenalDiseaseIndicator" in out.columns:
        out["has_renal_disease"] = out["RenalDiseaseIndicator"].astype(str) == "Y"
    return out


def build_health_claims_table(inpatient_df: pd.DataFrame, outpatient_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate inpatient + outpatient claims into one claim-level table with a claim_type flag."""
    ip = inpatient_df.copy()
    ip["claim_type"] = "inpatient"
    op = outpatient_df.copy()
    op["claim_type"] = "outpatient"
    combined = pd.concat([ip, op], ignore_index=True, sort=False)

    diag_cols = [c for c in HEALTH_DIAGNOSIS_CODE_COLUMNS if c in combined.columns]
    proc_cols = [c for c in HEALTH_PROCEDURE_CODE_COLUMNS if c in combined.columns]
    combined["num_diagnosis_codes"] = combined[diag_cols].notna().sum(axis=1)
    combined["num_procedure_codes"] = combined[proc_cols].notna().sum(axis=1)
    # Claim-level utilization/severity proxies available in both source tables.
    # Do not expose Provider or physician identifiers: the target is itself a
    # provider label, and identity memorization would invalidate group holdout.
    for col in ("InscClaimAmtReimbursed", "DeductibleAmtPaid"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    start = pd.to_datetime(combined["ClaimStartDt"], errors="coerce") if "ClaimStartDt" in combined else pd.Series(pd.NaT, index=combined.index)
    end = pd.to_datetime(combined["ClaimEndDt"], errors="coerce") if "ClaimEndDt" in combined else pd.Series(pd.NaT, index=combined.index)
    combined["claim_duration_days"] = (end - start).dt.days.clip(lower=0)
    if "AdmissionDt" in combined.columns:
        admission = pd.to_datetime(combined["AdmissionDt"], errors="coerce")
        combined["admission_to_claim_days"] = (start - admission).dt.days.clip(lower=0)
    return combined


def build_claim_fraud_label_provider_derived(
    claims_df: pd.DataFrame, provider_labels_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach the provider-derived claim-level fraud label to a claims table.

    This label is inherited from the claim's Provider, not verified per claim
    — see docs/data_contract.md section 11. The output column is named
    explicitly as `claim_fraud_label_provider_derived`, never `fraud`.
    """
    labels = provider_labels_df[[HEALTH_PROVIDER_COL, HEALTH_PROVIDER_LABEL_COL]].drop_duplicates()
    merged = claims_df.merge(labels, on=HEALTH_PROVIDER_COL, how="left", validate="many_to_one")
    merged[HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL] = (
        merged[HEALTH_PROVIDER_LABEL_COL] == "Yes"
    ).astype(int)
    return merged.drop(columns=[HEALTH_PROVIDER_LABEL_COL])
