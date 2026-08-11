"""Schema/quality validation for loaded raw dataframes.

Validation only inspects and reports; it never mutates the dataframe or
silently drops rows/columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.schemas import (
    DEMOGRAPHICS_COLUMNS,
    HEALTH_BENE_ID_COL,
    HEALTH_PROVIDER_COL,
    HEALTH_PROVIDER_LABEL_COL,
    MOTOR_ALL_COLUMNS,
    MOTOR_TARGET,
)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError(f"Validation failed: {'; '.join(self.errors)}")


def validate_columns(df: pd.DataFrame, expected_columns: list[str]) -> list[str]:
    """Return the subset of expected_columns missing from df."""
    return [c for c in expected_columns if c not in df.columns]


def validate_not_empty(df: pd.DataFrame) -> bool:
    return df.shape[0] > 0


def validate_target_present(df: pd.DataFrame, target_col: str) -> bool:
    return target_col in df.columns and df[target_col].notna().any()


def count_duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    return int(df.duplicated(subset=subset).sum())


def validate_motor(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = validate_columns(df, MOTOR_ALL_COLUMNS)
    if missing_cols:
        errors.append(f"missing expected columns: {missing_cols}")

    if not validate_not_empty(df):
        errors.append("dataframe is empty")

    if not validate_target_present(df, MOTOR_TARGET):
        errors.append(f"target column '{MOTOR_TARGET}' missing or entirely null")

    dup_count = count_duplicate_rows(df)
    if dup_count > 0:
        warnings.append(f"{dup_count} fully duplicate rows found")

    stats = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "duplicate_rows": dup_count,
        "missing_values_total": int(df.isna().sum().sum()),
    }
    if MOTOR_TARGET in df.columns:
        stats["target_positive_rate"] = float(df[MOTOR_TARGET].mean())

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, stats=stats)


def validate_health_main(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = validate_columns(df, [HEALTH_PROVIDER_COL])
    if missing_cols:
        errors.append(f"missing expected columns: {missing_cols}")

    if not validate_not_empty(df):
        errors.append("dataframe is empty")

    if HEALTH_PROVIDER_LABEL_COL in df.columns:
        if not validate_target_present(df, HEALTH_PROVIDER_LABEL_COL):
            errors.append(f"target column '{HEALTH_PROVIDER_LABEL_COL}' entirely null")
        unexpected = set(df[HEALTH_PROVIDER_LABEL_COL].dropna().unique()) - {"Yes", "No"}
        if unexpected:
            errors.append(f"unexpected values in '{HEALTH_PROVIDER_LABEL_COL}': {unexpected}")
    else:
        warnings.append(
            f"'{HEALTH_PROVIDER_LABEL_COL}' not present — this looks like the unlabeled "
            "Kaggle Test split, not usable for supervised training/evaluation"
        )

    dup_count = count_duplicate_rows(df)
    if dup_count > 0:
        warnings.append(f"{dup_count} fully duplicate rows found")

    stats = {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "duplicate_rows": dup_count}
    if HEALTH_PROVIDER_LABEL_COL in df.columns:
        stats["provider_positive_rate"] = float((df[HEALTH_PROVIDER_LABEL_COL] == "Yes").mean())

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, stats=stats)


def validate_health_claims(df: pd.DataFrame, expected_id_col: str = HEALTH_BENE_ID_COL) -> ValidationResult:
    """Validate an inpatient/outpatient claims table (shared shape)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not validate_not_empty(df):
        errors.append("dataframe is empty")

    missing_cols = validate_columns(df, [expected_id_col, "ClaimID", "Provider"])
    if missing_cols:
        errors.append(f"missing expected columns: {missing_cols}")

    if "ClaimID" in df.columns:
        dup_claim_ids = int(df["ClaimID"].duplicated().sum())
        if dup_claim_ids > 0:
            errors.append(f"{dup_claim_ids} duplicate ClaimID values found")

    stats = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, stats=stats)


def validate_demographics(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = validate_columns(df, DEMOGRAPHICS_COLUMNS)
    if missing_cols:
        errors.append(f"missing expected columns: {missing_cols}")

    if not validate_not_empty(df):
        errors.append("dataframe is empty")

    dup_count = count_duplicate_rows(df)
    if dup_count > 0:
        warnings.append(f"{dup_count} fully duplicate rows found")

    stats = {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "duplicate_rows": dup_count}
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, stats=stats)
