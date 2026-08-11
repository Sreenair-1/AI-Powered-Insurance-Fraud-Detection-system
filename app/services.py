"""Cached data loading, model fitting, and inference helpers for the Streamlit app.

Fits `HealthFraudModel` / `MotorFraudModel` in-process at app startup, cached
with `st.cache_resource` so it only runs once per server process — there is
no model-persistence layer in this repo (see docs/pipeline.md "Not
implemented here"; decided with the user for this work unit).

Demo-scale sampling
--------------------
Both domains' full graphs are too large for the exact centrality measures in
`src.graph.features` (`nx.betweenness_centrality` is O(V*E)) to compute
interactively:

- Health: ~558K claims -> a claim/provider/physician/beneficiary graph with
  well over a million edges. This module samples a stratified subset of
  `N_DEMO_HEALTH_PROVIDERS` providers (keeping the real per-claim label
  distribution) and caps claims per provider at `MAX_CLAIMS_PER_PROVIDER` so
  a few outlier high-volume providers can't blow up the graph size.
- Motor: 15,420 claims -> after synthetic GarageID/BankAccountID assignment,
  entity pool nodes become high-degree hubs that make betweenness far slower
  per node than health's graph shape. This module samples
  `N_DEMO_MOTOR_CLAIMS` rows from the *raw* motor claims *before* running
  synthetic augmentation, so the configured fraud rings are injected within
  the smaller sample (keeping full rings intact) rather than sampled away
  from a full-scale augmentation.

This is a demo/interactive-UI concession, not a change to the modeling
approach documented elsewhere in the repo — every function called here
(`fuse_motor_claims`, `fuse_health_claims`, `HealthFraudModel`,
`MotorFraudModel`, graph construction/features, SHAP, evaluation) is reused
unmodified. The UI labels every metric/count as computed on this demo
sample, not the full dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.data import preprocessing
from src.data.loaders import load_health_train_tables, load_motor_claims
from src.data.schemas import (
    HEALTH_BENE_ID_COL,
    HEALTH_PROVIDER_COL,
    HEALTH_PROVIDER_LABEL_COL,
    MOTOR_TARGET,
)
from src.data.splits import (
    assign_claims_to_provider_splits,
    split_dataframe,
    split_health_providers,
)
from src.data.synthetic import build_motor_synthetic_augmentation
from src.features.fusion import ForensicResultBundle, build_fused_features
from src.features.structured import (
    CLAIM_KEY_COL,
    build_health_structured_features,
    build_motor_structured_features,
)
from src.graph.construction import build_health_graph, build_motor_graph
from src.graph.features import extract_claim_features
from src.models.evaluation import ClassificationMetrics, evaluate_model
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.pipeline.inference import (
    ClaimRiskAssessment,
    DEFAULT_RISK_THRESHOLDS,
    RiskThresholds,
    assess_health_claims,
    assess_motor_claims,
    assess_single_health_claim,
    assess_single_motor_claim,
)

RANDOM_STATE = 42

# See module docstring for why these caps exist.
N_DEMO_MOTOR_CLAIMS = 1000
N_DEMO_HEALTH_PROVIDERS = 150
MAX_CLAIMS_PER_PROVIDER = 8


@dataclass(frozen=True)
class MotorAppData:
    model: MotorFraudModel
    graph_holdout: nx.Graph
    graph_features_holdout: pd.DataFrame
    structured_holdout: pd.DataFrame
    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    claims_holdout: pd.DataFrame
    claim_key_order: list[str]
    metrics: ClassificationMetrics
    assessments: dict[str, ClaimRiskAssessment]
    n_train: int


@dataclass(frozen=True)
class HealthAppData:
    model: HealthFraudModel
    graph_holdout: nx.Graph
    graph_features_holdout: pd.DataFrame
    structured_holdout: pd.DataFrame
    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    claims_holdout: pd.DataFrame
    beneficiary: pd.DataFrame
    claim_key_order: list[str]
    metrics: ClassificationMetrics
    assessments: dict[str, ClaimRiskAssessment]
    n_train: int
    n_providers_sampled: int


def _position_by_claim_key(claim_keys: list[str]) -> dict[str, int]:
    return {key: i for i, key in enumerate(claim_keys)}


@st.cache_resource(show_spinner="Loading motor claims, building the entity graph, and fitting the risk model...")
def load_motor_app_data() -> MotorAppData:
    """Build the demo-scale motor dataset, fit `MotorFraudModel`, and precompute holdout assessments."""
    motor_df = load_motor_claims()
    sample_size = min(N_DEMO_MOTOR_CLAIMS, len(motor_df))
    sample_df, _ = train_test_split(
        motor_df, train_size=sample_size, stratify=motor_df[MOTOR_TARGET], random_state=RANDOM_STATE
    )

    augmentation = build_motor_synthetic_augmentation(motor_df=sample_df, seed=RANDOM_STATE)
    claims = augmentation.claims

    train, val, test = split_dataframe(claims, stratify_col=MOTOR_TARGET, random_state=RANDOM_STATE)
    holdout = pd.concat([val, test]).sort_index()

    graph_train = build_motor_graph(train)
    X_train, y_train = _fuse_motor(train, graph_train)

    graph_holdout = build_motor_graph(holdout)
    structured_holdout = build_motor_structured_features(holdout)
    X_holdout, y_holdout = _fuse_motor(holdout, graph_holdout)

    model = MotorFraudModel().fit(X_train, y_train)
    metrics = evaluate_model(model, X_holdout, y_holdout)

    claim_key_order = list(X_holdout[CLAIM_KEY_COL])
    assessments = {
        a.claim_key: a for a in assess_motor_claims(model, X_holdout) if a.claim_key is not None
    }
    graph_features_holdout = extract_claim_features(graph_holdout)

    return MotorAppData(
        model=model,
        graph_holdout=graph_holdout,
        graph_features_holdout=graph_features_holdout,
        structured_holdout=structured_holdout,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        claims_holdout=holdout,
        claim_key_order=claim_key_order,
        metrics=metrics,
        assessments=assessments,
        n_train=len(train),
    )


def _fuse_motor(claims_df: pd.DataFrame, graph: nx.Graph):
    from src.features.fusion import fuse_motor_claims

    return fuse_motor_claims(claims_df, graph)


@st.cache_resource(show_spinner="Loading health claims, building the entity graph, and fitting the risk model...")
def load_health_app_data() -> HealthAppData:
    """Build the demo-scale health dataset, fit `HealthFraudModel`, and precompute holdout assessments."""
    from src.features.fusion import fuse_health_claims

    tables = load_health_train_tables()
    provider_labels = tables.main

    n_providers = min(N_DEMO_HEALTH_PROVIDERS, len(provider_labels))
    sampled_providers, _ = train_test_split(
        provider_labels,
        train_size=n_providers,
        stratify=provider_labels[HEALTH_PROVIDER_LABEL_COL],
        random_state=RANDOM_STATE,
    )
    provider_ids = set(sampled_providers[HEALTH_PROVIDER_COL])

    inpatient = _cap_claims_per_provider(
        tables.inpatient[tables.inpatient[HEALTH_PROVIDER_COL].isin(provider_ids)]
    )
    outpatient = _cap_claims_per_provider(
        tables.outpatient[tables.outpatient[HEALTH_PROVIDER_COL].isin(provider_ids)]
    )

    split_map = split_health_providers(
        sampled_providers,
        provider_col=HEALTH_PROVIDER_COL,
        label_col=HEALTH_PROVIDER_LABEL_COL,
        random_state=RANDOM_STATE,
    )
    ip_split = assign_claims_to_provider_splits(inpatient, HEALTH_PROVIDER_COL, split_map)
    op_split = assign_claims_to_provider_splits(outpatient, HEALTH_PROVIDER_COL, split_map)

    ip_train, op_train = inpatient[ip_split == "train"], outpatient[op_split == "train"]
    ip_hold = inpatient[ip_split.isin(["val", "test"])]
    op_hold = outpatient[op_split.isin(["val", "test"])]

    graph_train = build_health_graph(pd.concat([ip_train, op_train], ignore_index=True))
    X_train, y_train = fuse_health_claims(ip_train, op_train, tables.beneficiary, provider_labels, graph_train)

    graph_holdout = build_health_graph(pd.concat([ip_hold, op_hold], ignore_index=True))
    structured_holdout = build_health_structured_features(ip_hold, op_hold, tables.beneficiary)
    X_holdout, y_holdout = fuse_health_claims(
        ip_hold, op_hold, tables.beneficiary, provider_labels, graph_holdout
    )

    model = HealthFraudModel().fit(X_train, y_train)
    metrics = evaluate_model(model, X_holdout, y_holdout)

    claims_holdout = preprocessing.build_health_claims_table(ip_hold, op_hold)

    claim_key_order = list(X_holdout[CLAIM_KEY_COL])
    assessments = {
        a.claim_key: a for a in assess_health_claims(model, X_holdout) if a.claim_key is not None
    }
    graph_features_holdout = extract_claim_features(graph_holdout)

    return HealthAppData(
        model=model,
        graph_holdout=graph_holdout,
        graph_features_holdout=graph_features_holdout,
        structured_holdout=structured_holdout,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        claims_holdout=claims_holdout,
        beneficiary=tables.beneficiary,
        claim_key_order=claim_key_order,
        metrics=metrics,
        assessments=assessments,
        n_train=len(ip_train) + len(op_train),
        n_providers_sampled=n_providers,
    )


def _cap_claims_per_provider(df: pd.DataFrame, max_per_provider: int = MAX_CLAIMS_PER_PROVIDER) -> pd.DataFrame:
    """Bound each provider's claim count so a few high-volume providers can't blow up the graph."""
    sampled_positions = df.groupby(HEALTH_PROVIDER_COL).apply(
        lambda g: pd.Series(g.index).sample(n=min(len(g), max_per_provider), random_state=RANDOM_STATE)
    )
    return df.loc[sampled_positions.values].reset_index(drop=True)


def claim_row(claim_key: str, claim_key_order: list[str], raw_df: pd.DataFrame) -> pd.Series:
    """Look up the original (pre-fusion) claim record for `claim_key`.

    Fusion preserves row order/count from its input claims table (see
    `src.features.fusion`), so the position of `claim_key` in
    `claim_key_order` is the same position in `raw_df`.
    """
    position = _position_by_claim_key(claim_key_order)[claim_key]
    return raw_df.iloc[position]


def graph_claim_metrics(graph_features: pd.DataFrame, claim_key: str) -> Optional[dict[str, object]]:
    """Centrality/community/entity-sharing values for one claim node, or `None` if absent.

    `graph_features` is the precomputed `extract_claim_features()` table
    (`MotorAppData.graph_features_holdout` / `HealthAppData.graph_features_holdout`)
    — recomputing centrality per claim click would be too slow even at demo scale.
    """
    match = graph_features[graph_features["node"] == claim_key]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def reassess_motor_with_forensics(
    data: MotorAppData,
    claim_key: str,
    bundle: ForensicResultBundle,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
) -> ClaimRiskAssessment:
    """Re-run risk assessment for one motor claim with an uploaded document's forensic results attached."""
    structured_row = data.structured_holdout[data.structured_holdout[CLAIM_KEY_COL] == claim_key]
    X_row = build_fused_features(structured_row, data.graph_holdout, forensic_results={claim_key: bundle})
    return assess_single_motor_claim(data.model, X_row, thresholds=thresholds)


def reassess_health_with_forensics(
    data: HealthAppData,
    claim_key: str,
    bundle: ForensicResultBundle,
    thresholds: RiskThresholds = DEFAULT_RISK_THRESHOLDS,
) -> ClaimRiskAssessment:
    """Re-run risk assessment for one health claim with an uploaded document's forensic results attached."""
    structured_row = data.structured_holdout[data.structured_holdout[CLAIM_KEY_COL] == claim_key]
    X_row = build_fused_features(structured_row, data.graph_holdout, forensic_results={claim_key: bundle})
    return assess_single_health_claim(data.model, X_row, thresholds=thresholds)
