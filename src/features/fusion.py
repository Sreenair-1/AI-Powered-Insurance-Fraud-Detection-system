"""Feature fusion — the final model-ready, claim-keyed feature matrix.

Joins the three adapter outputs (`src.features.structured`,
`src.features.graph`, `src.features.forensic`) into one `claim_key`-indexed
`pandas.DataFrame`, with the training target returned separately (never as
a feature column).

This module fits nothing (no encoder/scaler) and reuses the existing
adapters/preprocessing functions rather than re-deriving any feature —
it is purely a join/assembly layer. It never mutates any input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import networkx as nx
import pandas as pd

from src.data import preprocessing
from src.data.schemas import (
    HEALTH_CLAIM_ID_COL,
    HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL,
    MOTOR_TARGET,
)
from src.features.forensic import (
    FORENSIC_FEATURE_NAMES,
    CopyMoveResult,
    ELAResult,
    MetadataResult,
    extract_forensic_features,
)
from src.features.graph import build_graph_claim_features
from src.features.structured import (
    CLAIM_KEY_COL,
    build_health_structured_features,
    build_motor_structured_features,
)
from src.graph.construction import CLAIM_NODE_TYPE

TARGET_COL = "target"

# Columns that must never appear in a fused feature matrix, regardless of
# whether an upstream adapter already stripped them — a fused matrix is the
# final thing a model sees, so this is checked again here defensively.
_LEAKAGE_COLUMNS = frozenset(
    {
        MOTOR_TARGET,
        HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL,
        "synthetic_fraud_ring_label",
        "synthetic_ring_id",
    }
)


def _claim_key(raw_id: object) -> str:
    """Build the `"claim:{raw_id}"` join key, matching `src.graph.construction`."""
    return f"{CLAIM_NODE_TYPE}:{raw_id}"


def _drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in _LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=present) if present else df


@dataclass(frozen=True)
class ForensicResultBundle:
    """The forensic check results available for a single claim's document.

    Any field may be `None` if that particular check was not run or no
    document was attached — `build_forensic_claim_features` / `extract_
    forensic_features` already handle `None` as "unavailable".
    """

    ela: Optional[ELAResult] = None
    copy_move: Optional[CopyMoveResult] = None
    metadata: Optional[MetadataResult] = None


def build_forensic_claim_features(
    claim_keys: Iterable[str],
    forensic_results: Optional[Mapping[str, ForensicResultBundle]] = None,
) -> pd.DataFrame:
    """Build a claim-keyed forensic feature table aligned to `claim_keys`.

    Args:
        claim_keys: The full ordered set of `claim_key` values to produce a
            row for.
        forensic_results: Optional mapping of `claim_key` ->
            `ForensicResultBundle`. A `claim_key` absent from this mapping
            (or `forensic_results is None` entirely) gets the same
            "unavailable" placeholders `extract_forensic_features()`
            already returns for `None` inputs — most claims have no
            attached document, and that must not produce `NaN`-shaped
            surprises anywhere but the forensic columns themselves.

    Returns:
        One row per entry of `claim_keys`, columns `claim_key` plus every
        column in `FORENSIC_FEATURE_NAMES`.
    """
    forensic_results = forensic_results or {}
    claim_keys_list = list(claim_keys)
    rows = []
    for key in claim_keys_list:
        bundle = forensic_results.get(key)
        if bundle is None:
            row = extract_forensic_features()
        else:
            row = extract_forensic_features(bundle.ela, bundle.copy_move, bundle.metadata)
        row[CLAIM_KEY_COL] = key
        rows.append(row)
    return pd.DataFrame(rows, columns=[CLAIM_KEY_COL, *FORENSIC_FEATURE_NAMES]).reset_index(drop=True)


def build_fused_features(
    structured_df: pd.DataFrame,
    graph: nx.Graph,
    forensic_results: Optional[Mapping[str, ForensicResultBundle]] = None,
    claim_node_type: str = CLAIM_NODE_TYPE,
) -> pd.DataFrame:
    """Join structured + graph + forensic features into one claim-keyed matrix.

    Args:
        structured_df: Output of `build_motor_structured_features` or
            `build_health_structured_features` (must carry `claim_key`,
            one row per claim, no duplicates).
        graph: The corresponding `networkx.Graph` from `build_health_graph`
            or `build_motor_graph`. Never mutated.
        forensic_results: Optional `claim_key` -> `ForensicResultBundle`
            mapping; claims with no entry get "unavailable" forensic
            features (see `build_forensic_claim_features`).
        claim_node_type: Passed through to `build_graph_claim_features`.

    Returns:
        `structured_df`, left-joined with the aligned graph and forensic
        feature tables on `claim_key`, with any known label/leakage column
        stripped defensively. Never contains `FraudFound_P`,
        `claim_fraud_label_provider_derived`, `synthetic_fraud_ring_label`,
        or `synthetic_ring_id`. Row order and count match `structured_df`.

    Raises:
        ValueError: if `structured_df` has no `claim_key` column, or has
            duplicate `claim_key` values (fusion requires exactly one row
            per claim so the join can't silently fan out rows).
    """
    if CLAIM_KEY_COL not in structured_df.columns:
        raise ValueError(f"structured_df must contain a '{CLAIM_KEY_COL}' column")

    structured = _drop_leakage_columns(structured_df).reset_index(drop=True)
    claim_keys = list(structured[CLAIM_KEY_COL])
    if len(set(claim_keys)) != len(claim_keys):
        raise ValueError(
            "structured_df has duplicate claim_key values; fusion requires exactly one row per claim"
        )

    graph_df = build_graph_claim_features(graph, claim_keys, claim_node_type=claim_node_type)
    forensic_df = build_forensic_claim_features(claim_keys, forensic_results)

    fused = structured.merge(graph_df, on=CLAIM_KEY_COL, how="left", validate="one_to_one")
    fused = fused.merge(forensic_df, on=CLAIM_KEY_COL, how="left", validate="one_to_one")
    return _drop_leakage_columns(fused)


def build_motor_target(claims_df: pd.DataFrame) -> pd.DataFrame:
    """Build a claim-keyed `FraudFound_P` target table for motor claims.

    Uses the same row-index-based `claim_key` scheme as
    `build_motor_structured_features`, so the result can be aligned to a
    fused motor feature matrix on `claim_key`. Does not mutate `claims_df`.
    """
    y = claims_df[MOTOR_TARGET].astype(int)
    return pd.DataFrame(
        {
            CLAIM_KEY_COL: [_claim_key(idx) for idx in claims_df.index],
            TARGET_COL: y.to_numpy(),
        }
    ).reset_index(drop=True)


def build_health_target(
    inpatient_df: pd.DataFrame,
    outpatient_df: pd.DataFrame,
    provider_labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a claim-keyed provider-derived target table for health claims.

    Wraps `preprocessing.build_health_claims_table` +
    `preprocessing.build_claim_fraud_label_provider_derived` and attaches
    the same `ClaimID`-based `claim_key` scheme as
    `build_health_structured_features`. Does not mutate any input. See
    `docs/data_contract.md` section 11 for the provider-derived-label
    caveat: this is not a verified per-claim label.
    """
    claims = preprocessing.build_health_claims_table(inpatient_df, outpatient_df)
    labeled = preprocessing.build_claim_fraud_label_provider_derived(claims, provider_labels_df)
    return pd.DataFrame(
        {
            CLAIM_KEY_COL: [_claim_key(cid) for cid in labeled[HEALTH_CLAIM_ID_COL]],
            TARGET_COL: labeled[HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL].to_numpy(),
        }
    ).reset_index(drop=True)


def _align_target(X: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    merged = X[[CLAIM_KEY_COL]].merge(target_df, on=CLAIM_KEY_COL, how="left", validate="one_to_one")
    if merged[TARGET_COL].isna().any():
        raise ValueError(
            "target could not be aligned for every claim_key in the feature matrix "
            "(check that the same underlying claims data was used for both)"
        )
    y = merged[TARGET_COL].astype(int)
    y.index = X.index
    y.name = TARGET_COL
    return y


def fuse_motor_claims(
    claims_df: pd.DataFrame,
    graph: nx.Graph,
    forensic_results: Optional[Mapping[str, ForensicResultBundle]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full motor feature matrix and its aligned target.

    `claims_df` must be the same augmented motor claims DataFrame (same row
    index) used to build `graph` via `build_motor_graph` /
    `build_motor_synthetic_augmentation`, otherwise `claim_key` values won't
    line up between the structured and graph sides.

    Returns:
        `(X, y)` — `X` is the fused, claim-keyed feature matrix (no target
        or leakage columns); `y` is the `FraudFound_P` target as a `pandas.
        Series` aligned row-for-row with `X` (same index, not merged into
        `X`).
    """
    structured_df = build_motor_structured_features(claims_df)
    X = build_fused_features(structured_df, graph, forensic_results)
    y = _align_target(X, build_motor_target(claims_df))
    return X, y


def fuse_health_claims(
    inpatient_df: pd.DataFrame,
    outpatient_df: pd.DataFrame,
    beneficiary_df: pd.DataFrame,
    provider_labels_df: pd.DataFrame,
    graph: nx.Graph,
    forensic_results: Optional[Mapping[str, ForensicResultBundle]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full health feature matrix and its aligned target.

    `inpatient_df`/`outpatient_df` must be the same tables (by `ClaimID`)
    used to build `graph` via `build_health_graph`, otherwise `claim_key`
    values won't line up between the structured and graph sides.

    Returns:
        `(X, y)` — `X` is the fused, claim-keyed feature matrix (no target
        or leakage columns); `y` is the provider-derived target as a
        `pandas.Series` aligned row-for-row with `X` (same index, not
        merged into `X`). See `docs/data_contract.md` section 11: this
        target is a provider-level label inherited onto claims, not a
        verified per-claim label.
    """
    structured_df = build_health_structured_features(inpatient_df, outpatient_df, beneficiary_df)
    X = build_fused_features(structured_df, graph, forensic_results)
    y = _align_target(X, build_health_target(inpatient_df, outpatient_df, provider_labels_df))
    return X, y
