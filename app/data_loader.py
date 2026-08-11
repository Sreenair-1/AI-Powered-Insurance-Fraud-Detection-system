"""Sampled dataset building for the Streamlit app.

Builds the same claim-keyed, fused feature matrices `src/models/` and
`src/pipeline/inference.py` expect (`src.features.fusion`), over a bounded,
seeded sample of the real raw data (see `app/config.py` for why sampling is
necessary). Also keeps the intermediate `structured_df` around (not just the
final fused `X`) so a single claim can be re-fused with a real forensic
bundle later (see `app/claim_view.py`'s `rescore_with_forensics`) without
rebuilding the whole dataset.

Every public function here is read-only over the raw data on disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import pandas as pd
import streamlit as st

from src.data import preprocessing
from src.data.loaders import DatasetFileMissingError, load_health_train_tables, load_motor_claims
from src.data.schemas import HEALTH_CLAIM_ID_COL
from src.data.synthetic import build_motor_synthetic_augmentation
from src.features.fusion import (
    build_fused_features,
    build_health_target,
    build_motor_target,
    TARGET_COL,
)
from src.features.structured import (
    CLAIM_KEY_COL,
    build_health_structured_features,
    build_motor_structured_features,
)
from src.graph.construction import build_health_graph, build_motor_graph

from app.config import (
    DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE,
    DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE,
    DEFAULT_MOTOR_SAMPLE_SIZE,
    SAMPLE_SEED,
)


class DatasetNotAvailableError(RuntimeError):
    """Raised when the raw claims data required to build the app dataset is missing."""


@dataclass(frozen=True)
class MotorDataset:
    claims: pd.DataFrame  # augmented, sampled claims (original row index preserved)
    structured_df: pd.DataFrame
    graph: nx.Graph
    X: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class HealthDataset:
    claims: pd.DataFrame  # combined inpatient+outpatient sampled claims table
    structured_df: pd.DataFrame
    graph: nx.Graph
    X: pd.DataFrame
    y: pd.Series


def _sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _align_target(X: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series:
    """Duplicated from `src.features.fusion._align_target` (private, not part of its public API).

    Needed here because `fuse_motor_claims`/`fuse_health_claims` don't expose
    the intermediate `structured_df` this module also needs for
    forensic-upload rescoring, so the target alignment step is redone
    directly on `build_fused_features` output instead of going through them.
    """
    merged = X[[CLAIM_KEY_COL]].merge(target_df, on=CLAIM_KEY_COL, how="left", validate="one_to_one")
    if merged[TARGET_COL].isna().any():
        raise DatasetNotAvailableError(
            "target could not be aligned for every claim in the sampled dataset "
            "(this should not happen -- please report it)"
        )
    y = merged[TARGET_COL].astype(int)
    y.index = X.index
    y.name = TARGET_COL
    return y


def build_motor_dataset(
    sample_size: int = DEFAULT_MOTOR_SAMPLE_SIZE, seed: int = SAMPLE_SEED
) -> MotorDataset:
    """Load, sample, augment, and fuse the motor claims dataset."""
    try:
        raw = load_motor_claims()
    except DatasetFileMissingError as exc:
        raise DatasetNotAvailableError(f"Motor claims data not found: {exc}") from exc

    sampled = _sample(raw, sample_size, seed)
    augmentation = build_motor_synthetic_augmentation(motor_df=sampled, seed=seed)
    claims = augmentation.claims

    graph = build_motor_graph(claims)
    structured_df = build_motor_structured_features(claims)
    X = build_fused_features(structured_df, graph)
    y = _align_target(X, build_motor_target(claims))
    return MotorDataset(claims=claims, structured_df=structured_df, graph=graph, X=X, y=y)


def build_health_dataset(
    inpatient_sample_size: int = DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE,
    outpatient_sample_size: int = DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> HealthDataset:
    """Load, sample, and fuse the health claims dataset."""
    try:
        tables = load_health_train_tables()
    except DatasetFileMissingError as exc:
        raise DatasetNotAvailableError(f"Health claims data not found: {exc}") from exc

    inpatient = _sample(tables.inpatient, inpatient_sample_size, seed)
    outpatient = _sample(tables.outpatient, outpatient_sample_size, seed)
    claims = preprocessing.build_health_claims_table(inpatient, outpatient)

    graph = build_health_graph(claims)
    structured_df = build_health_structured_features(inpatient, outpatient, tables.beneficiary)
    X = build_fused_features(structured_df, graph)
    y = _align_target(X, build_health_target(inpatient, outpatient, tables.main))
    return HealthDataset(claims=claims, structured_df=structured_df, graph=graph, X=X, y=y)


@st.cache_data(show_spinner="Loading motor claims and building features...")
def get_cached_motor_dataset(
    sample_size: int = DEFAULT_MOTOR_SAMPLE_SIZE, seed: int = SAMPLE_SEED
) -> MotorDataset:
    return build_motor_dataset(sample_size, seed)


@st.cache_data(show_spinner="Loading health claims and building features...")
def get_cached_health_dataset(
    inpatient_sample_size: int = DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE,
    outpatient_sample_size: int = DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> HealthDataset:
    return build_health_dataset(inpatient_sample_size, outpatient_sample_size, seed)


def motor_claim_row(dataset: MotorDataset, claim_key: str) -> pd.Series:
    """The raw (augmented) claim row for `claim_key`, e.g. `\"claim:1234\"`."""
    raw_index = claim_key.split(":", 1)[1]
    try:
        row_label: object = int(raw_index)
    except ValueError:
        row_label = raw_index
    try:
        return dataset.claims.loc[row_label]
    except KeyError as exc:
        raise DatasetNotAvailableError(f"claim_key {claim_key!r} not found in loaded motor claims") from exc


def health_claim_row(dataset: HealthDataset, claim_key: str) -> pd.Series:
    """The raw claim row for `claim_key`, e.g. `\"claim:CLM12345\"`."""
    claim_id = claim_key.split(":", 1)[1]
    matches = dataset.claims[dataset.claims[HEALTH_CLAIM_ID_COL] == claim_id]
    if matches.empty:
        raise DatasetNotAvailableError(f"claim_key {claim_key!r} not found in loaded health claims")
    return matches.iloc[0]
