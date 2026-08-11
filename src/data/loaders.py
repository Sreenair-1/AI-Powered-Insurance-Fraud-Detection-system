"""Raw dataset loaders. Read-only: never mutate or move files under data/raw/."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.schemas import (
    DEMOGRAPHICS_RAW_PATH,
    HEALTH_TEST_BENEFICIARY_PATH,
    HEALTH_TEST_INPATIENT_PATH,
    HEALTH_TEST_MAIN_PATH,
    HEALTH_TEST_OUTPATIENT_PATH,
    HEALTH_TRAIN_BENEFICIARY_PATH,
    HEALTH_TRAIN_INPATIENT_PATH,
    HEALTH_TRAIN_MAIN_PATH,
    HEALTH_TRAIN_OUTPATIENT_PATH,
    MOTOR_RAW_PATH,
)


class DatasetFileMissingError(FileNotFoundError):
    """Raised when an expected raw dataset file is not present on disk."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DatasetFileMissingError(
            f"Expected dataset file not found: {path}. "
            "See docs/data_contract.md section 2 for the required local file "
            "layout under data/raw/."
        )
    return pd.read_csv(path, low_memory=False)


def load_motor_claims() -> pd.DataFrame:
    """Load the motor claims dataset (fraud_oracle.csv)."""
    return _read_csv(MOTOR_RAW_PATH)


def load_demographics_reference() -> pd.DataFrame:
    """Load the Indian demographics reference dataset.

    Reference distribution only — never join directly with claims data
    (see docs/data_contract.md section 12).
    """
    return _read_csv(DEMOGRAPHICS_RAW_PATH)


@dataclass(frozen=True)
class HealthRawTables:
    main: pd.DataFrame
    beneficiary: pd.DataFrame
    inpatient: pd.DataFrame
    outpatient: pd.DataFrame


def load_health_train_tables() -> HealthRawTables:
    """Load the labeled health Train tables (main + beneficiary + inpatient + outpatient).

    This is the only health split usable for supervised training/evaluation —
    Kaggle's Test* files carry no PotentialFraud label at all (see
    docs/data_contract.md section 3).
    """
    return HealthRawTables(
        main=_read_csv(HEALTH_TRAIN_MAIN_PATH),
        beneficiary=_read_csv(HEALTH_TRAIN_BENEFICIARY_PATH),
        inpatient=_read_csv(HEALTH_TRAIN_INPATIENT_PATH),
        outpatient=_read_csv(HEALTH_TRAIN_OUTPATIENT_PATH),
    )


def load_health_test_tables() -> HealthRawTables:
    """Load the unlabeled health Test tables.

    `main` here has only a `Provider` column (no PotentialFraud). Not usable
    for supervised evaluation — see docs/data_contract.md section 3.
    """
    return HealthRawTables(
        main=_read_csv(HEALTH_TEST_MAIN_PATH),
        beneficiary=_read_csv(HEALTH_TEST_BENEFICIARY_PATH),
        inpatient=_read_csv(HEALTH_TEST_INPATIENT_PATH),
        outpatient=_read_csv(HEALTH_TEST_OUTPATIENT_PATH),
    )
