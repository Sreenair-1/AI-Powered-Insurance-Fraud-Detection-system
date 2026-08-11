"""Leakage-safe, deterministic train/validation/test splitting.

Motor claims are independent rows and are split directly. Health claims are
split by Provider (see docs/data_contract.md section 10): PotentialFraud is
provider-level, so splitting individual claim rows at random would leak the
label across train/val/test through provider identity.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def split_dataframe(
    df: pd.DataFrame,
    stratify_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split on stratify_col. Row-independent data only."""
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be in (0, 1)")

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[stratify_col],
        random_state=random_state,
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[stratify_col],
        random_state=random_state,
    )
    return train, val, test


def split_health_providers(
    provider_labels: pd.DataFrame,
    provider_col: str,
    label_col: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> dict[str, str]:
    """Split providers (not claims) into train/val/test, stratified on the provider label.

    Returns a mapping {provider_id: "train" | "val" | "test"}.
    """
    train, val, test = split_dataframe(
        provider_labels, stratify_col=label_col, val_size=val_size, test_size=test_size,
        random_state=random_state,
    )
    mapping: dict[str, str] = {}
    for split_name, split_df in (("train", train), ("val", val), ("test", test)):
        for provider_id in split_df[provider_col]:
            mapping[provider_id] = split_name
    return mapping


def assign_claims_to_provider_splits(
    claims_df: pd.DataFrame,
    provider_col: str,
    provider_split_map: dict[str, str],
) -> pd.Series:
    """Map each claim row to its provider's split assignment ('train'/'val'/'test').

    Raises if any claim's provider is not present in provider_split_map, since
    that would silently drop claims from the split.
    """
    unknown = set(claims_df[provider_col]) - set(provider_split_map)
    if unknown:
        raise ValueError(
            f"{len(unknown)} providers in claims_df have no split assignment "
            f"(e.g. {sorted(unknown)[:5]})"
        )
    return claims_df[provider_col].map(provider_split_map)
