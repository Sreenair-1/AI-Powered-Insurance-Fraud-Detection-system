"""Structured feature adapter.

Wraps `src.data.preprocessing` outputs into clean, claim-keyed feature
tables ready for feature fusion. Every row carries a `claim_key` column in
the same `"claim:{raw_id}"` format used by `src.graph.construction` node
ids (`"claim:{row_index}"` for motor, `"claim:{ClaimID}"` for health), so
structured and graph features can be joined without re-deriving the key
scheme.

This module never fits any encoder/scaler and never touches
`FraudFound_P` or `claim_fraud_label_provider_derived` except to strip them
out defensively if a caller's input already carries them — those columns
are training targets, never model inputs here.
"""
from __future__ import annotations

import pandas as pd

from src.data import preprocessing
from src.data.schemas import (
    HEALTH_BENE_ID_COL,
    HEALTH_CLAIM_ID_COL,
    HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL,
    MOTOR_TARGET,
)
from src.graph.construction import CLAIM_NODE_TYPE

CLAIM_KEY_COL = "claim_key"

# Columns that must never appear in a structured feature matrix, regardless
# of whether the caller's raw input already carries them.
_LEAKAGE_COLUMNS = frozenset(
    {
        MOTOR_TARGET,
        HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL,
        "synthetic_fraud_ring_label",
        "synthetic_ring_id",
        # These two columns are invented by the demo augmentation and have
        # no counterpart in the observed motor claims dataset.
        "GarageID",
        "BankAccountID",
    }
)

_HEALTH_BENEFICIARY_FEATURE_COLUMNS = ["is_deceased", "chronic_condition_count", "has_renal_disease"]


def _claim_key(raw_id: object) -> str:
    """Build the `"claim:{raw_id}"` join key shared with graph node ids."""
    return f"{CLAIM_NODE_TYPE}:{raw_id}"


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in _LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=present) if present else df


def build_motor_structured_features(claims_df: pd.DataFrame) -> pd.DataFrame:
    """Build claim-keyed structured features for motor claims.

    Reuses `preprocessing.build_motor_feature_target` for identifier
    dropping and categorical encoding, discards the target, and attaches a
    `claim_key` column derived from `claims_df`'s row index — matching
    `src.graph.construction.build_motor_graph`, which also keys motor claim
    nodes by row index (motor has no native `ClaimID`).

    Does not mutate `claims_df`. Deterministic for a given input.
    """
    X, _ = preprocessing.build_motor_feature_target(claims_df)
    X = _drop_leakage_columns(X)
    out = X.copy()
    out.insert(0, CLAIM_KEY_COL, [_claim_key(idx) for idx in claims_df.index])
    return out.reset_index(drop=True)


def build_health_structured_features(
    inpatient_df: pd.DataFrame,
    outpatient_df: pd.DataFrame,
    beneficiary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build claim-keyed structured features for health claims.

    Combines `preprocessing.build_health_claims_table()` (`claim_type`,
    `num_diagnosis_codes`, `num_procedure_codes`) with beneficiary-derived
    features from `preprocessing.add_beneficiary_derived_features()`
    (`is_deceased`, `chronic_condition_count`, `has_renal_disease`), joined
    on `BeneID`. Attaches `claim_key` (`"claim:{ClaimID}"`), matching
    `src.graph.construction.build_health_graph`'s claim node id scheme.

    Raw diagnosis/procedure codes, dates, and identifier columns are left
    out of the returned feature set — this adapter only surfaces the
    engineered columns `preprocessing.py` already produces, not new
    feature engineering. Never includes
    `claim_fraud_label_provider_derived` (callers attach labels
    separately, only for training). Does not mutate any input. Beneficiary
    rows are deduplicated on `BeneID` before the join, so a beneficiary
    appearing multiple times in `beneficiary_df` cannot fan out claim rows.
    """
    claims = preprocessing.build_health_claims_table(inpatient_df, outpatient_df)
    beneficiary = preprocessing.add_beneficiary_derived_features(beneficiary_df)

    bene_cols = [HEALTH_BENE_ID_COL] + [
        c for c in _HEALTH_BENEFICIARY_FEATURE_COLUMNS if c in beneficiary.columns
    ]
    beneficiary_features = beneficiary[bene_cols].drop_duplicates(subset=[HEALTH_BENE_ID_COL])

    merged = claims.merge(beneficiary_features, on=HEALTH_BENE_ID_COL, how="left", validate="many_to_one")

    feature_cols = [
        HEALTH_CLAIM_ID_COL,
        "claim_type",
        "num_diagnosis_codes",
        "num_procedure_codes",
    ] + [c for c in _HEALTH_BENEFICIARY_FEATURE_COLUMNS if c in merged.columns]
    out = merged[feature_cols].copy()
    out = _drop_leakage_columns(out)
    claim_keys = [_claim_key(cid) for cid in out[HEALTH_CLAIM_ID_COL]]
    out = out.drop(columns=[HEALTH_CLAIM_ID_COL])
    if "claim_type" in out.columns:
        out["claim_type"] = out["claim_type"].astype("category")
    out.insert(0, CLAIM_KEY_COL, claim_keys)
    return out.reset_index(drop=True)
