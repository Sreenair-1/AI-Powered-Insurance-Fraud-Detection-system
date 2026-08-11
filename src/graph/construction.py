"""Graph construction for health and motor claim entity networks.

Builds a `networkx.Graph` connecting claims to the surrounding entities:

- **Health**: `Provider` / `AttendingPhysician` / `BeneID` are real columns
  from the health raw data (see `src/data/schemas.py`). Only the attending
  physician is modeled as the "Physician" node type (OperatingPhysician/
  OtherPhysician are frequently missing and are out of scope here).
- **Motor**: `GarageID` / `BankAccountID` do not exist in the real motor
  dataset — they only exist as output of
  `src.data.synthetic.build_motor_synthetic_augmentation()`. This module
  never invents them itself; callers must pass already-augmented claims.

Every node id is prefixed with its type (e.g. `"claim:CLM12345"`,
`"garage:GRG000001"`) so ids never collide across node types sharing the same
raw value space. Building a graph never mutates the input DataFrame.
"""
from __future__ import annotations

from typing import Hashable

import networkx as nx
import pandas as pd

from src.data.schemas import (
    HEALTH_BENE_ID_COL,
    HEALTH_CLAIM_ID_COL,
    HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL,
    HEALTH_PROVIDER_COL,
)
from src.data.synthetic import (
    BANK_ACCOUNT_ID_COL,
    GARAGE_ID_COL,
    RING_ID_COL,
    RING_LABEL_COL,
)

# Not in schemas.py as a named constant (only listed inside
# HEALTH_CLAIM_SHARED_COLUMNS) — named locally since this is the only module
# that needs it as a standalone symbol.
HEALTH_ATTENDING_PHYSICIAN_COL = "AttendingPhysician"

MOTOR_TARGET_COL = "FraudFound_P"

NODE_TYPE_ATTR = "node_type"
EDGE_TYPE_ATTR = "edge_type"

CLAIM_NODE_TYPE = "claim"
PROVIDER_NODE_TYPE = "provider"
PHYSICIAN_NODE_TYPE = "physician"
BENEFICIARY_NODE_TYPE = "beneficiary"
GARAGE_NODE_TYPE = "garage"
BANK_ACCOUNT_NODE_TYPE = "bank_account"


def _node_id(node_type: str, raw_id: Hashable) -> str:
    """Build a collision-safe node id, e.g. `"provider:PRV51001"`."""
    return f"{node_type}:{raw_id}"


def _require_columns(df: pd.DataFrame, required: set[str], caller: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{caller}: claims_df is missing required columns {sorted(missing)}")


def build_health_graph(claims_df: pd.DataFrame) -> nx.Graph:
    """Build the health claim entity graph.

    Node types: `claim`, `provider`, `physician`, `beneficiary`.
    Edges connect each claim to its provider, beneficiary, and (when present)
    attending physician. `claims_df` is read-only and is never modified;
    rows with a missing `AttendingPhysician` simply produce no physician
    edge for that claim.
    """
    required = {
        HEALTH_CLAIM_ID_COL,
        HEALTH_PROVIDER_COL,
        HEALTH_BENE_ID_COL,
        HEALTH_ATTENDING_PHYSICIAN_COL,
    }
    _require_columns(claims_df, required, "build_health_graph")

    graph = nx.Graph()
    has_label_col = HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL in claims_df.columns
    has_claim_type_col = "claim_type" in claims_df.columns

    for _, row in claims_df.iterrows():
        claim_raw_id = row[HEALTH_CLAIM_ID_COL]
        claim_node = _node_id(CLAIM_NODE_TYPE, claim_raw_id)
        claim_attrs = {NODE_TYPE_ATTR: CLAIM_NODE_TYPE, "claim_id": claim_raw_id}
        if has_claim_type_col:
            claim_attrs["claim_type"] = row["claim_type"]
        if has_label_col:
            claim_attrs[HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL] = row[
                HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL
            ]
        graph.add_node(claim_node, **claim_attrs)

        provider_raw_id = row[HEALTH_PROVIDER_COL]
        provider_node = _node_id(PROVIDER_NODE_TYPE, provider_raw_id)
        graph.add_node(
            provider_node, **{NODE_TYPE_ATTR: PROVIDER_NODE_TYPE, "provider_id": provider_raw_id}
        )
        graph.add_edge(claim_node, provider_node, **{EDGE_TYPE_ATTR: "claim_provider"})

        bene_raw_id = row[HEALTH_BENE_ID_COL]
        bene_node = _node_id(BENEFICIARY_NODE_TYPE, bene_raw_id)
        graph.add_node(
            bene_node, **{NODE_TYPE_ATTR: BENEFICIARY_NODE_TYPE, "bene_id": bene_raw_id}
        )
        graph.add_edge(claim_node, bene_node, **{EDGE_TYPE_ATTR: "claim_beneficiary"})

        physician_raw_id = row[HEALTH_ATTENDING_PHYSICIAN_COL]
        if pd.notna(physician_raw_id):
            physician_node = _node_id(PHYSICIAN_NODE_TYPE, physician_raw_id)
            graph.add_node(
                physician_node,
                **{NODE_TYPE_ATTR: PHYSICIAN_NODE_TYPE, "physician_id": physician_raw_id},
            )
            graph.add_edge(claim_node, physician_node, **{EDGE_TYPE_ATTR: "claim_physician"})

    return graph


def build_motor_graph(claims_df: pd.DataFrame) -> nx.Graph:
    """Build the motor claim entity graph.

    Node types: `claim`, `garage`, `bank_account`. `claims_df` must already
    carry `GarageID`/`BankAccountID` — i.e. it must be the output of
    `src.data.synthetic.assign_synthetic_entities()` /
    `build_motor_synthetic_augmentation()`, never the raw motor dataset.
    The claim node id is the DataFrame row index (there is no native
    `ClaimID` column in the motor dataset). `claims_df` is read-only and is
    never modified.
    """
    required = {GARAGE_ID_COL, BANK_ACCOUNT_ID_COL}
    _require_columns(claims_df, required, "build_motor_graph")

    graph = nx.Graph()
    has_target_col = MOTOR_TARGET_COL in claims_df.columns
    has_ring_label_col = RING_LABEL_COL in claims_df.columns
    has_ring_id_col = RING_ID_COL in claims_df.columns

    for row_label, row in claims_df.iterrows():
        claim_node = _node_id(CLAIM_NODE_TYPE, row_label)
        claim_attrs = {NODE_TYPE_ATTR: CLAIM_NODE_TYPE, "claim_row_index": row_label}
        if has_target_col:
            claim_attrs[MOTOR_TARGET_COL] = row[MOTOR_TARGET_COL]
        if has_ring_label_col:
            claim_attrs[RING_LABEL_COL] = row[RING_LABEL_COL]
        if has_ring_id_col:
            claim_attrs[RING_ID_COL] = row[RING_ID_COL]
        graph.add_node(claim_node, **claim_attrs)

        garage_raw_id = row[GARAGE_ID_COL]
        garage_node = _node_id(GARAGE_NODE_TYPE, garage_raw_id)
        graph.add_node(
            garage_node, **{NODE_TYPE_ATTR: GARAGE_NODE_TYPE, "garage_id": garage_raw_id}
        )
        graph.add_edge(claim_node, garage_node, **{EDGE_TYPE_ATTR: "claim_garage"})

        bank_raw_id = row[BANK_ACCOUNT_ID_COL]
        bank_node = _node_id(BANK_ACCOUNT_NODE_TYPE, bank_raw_id)
        graph.add_node(
            bank_node, **{NODE_TYPE_ATTR: BANK_ACCOUNT_NODE_TYPE, "bank_account_id": bank_raw_id}
        )
        graph.add_edge(claim_node, bank_node, **{EDGE_TYPE_ATTR: "claim_bank_account"})

    return graph
