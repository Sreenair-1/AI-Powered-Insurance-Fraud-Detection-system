"""Unit tests for app/data_loader.py.

Tiny synthetic/in-memory fixtures only (monkeypatches the loader functions
`app.data_loader` calls), mirroring tests/test_loaders.py's pattern -- no
data/raw/ dependency, so this suite runs in CI.

Motor fixtures use >=100 rows: `src.data.synthetic.build_motor_synthetic_augmentation`'s
default `FraudRingConfig` (5 rings x 4 claims = 20) needs enough claims that
20 stays under its 25% `max_affected_proportion` safety cap.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app import data_loader
from src.data.loaders import DatasetFileMissingError, HealthRawTables
from src.data.schemas import MOTOR_ALL_COLUMNS, MOTOR_TARGET
from src.features.structured import CLAIM_KEY_COL


def _tiny_motor_claims(n: int = 150) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {c: 0 for c in MOTOR_ALL_COLUMNS}
        row["Make"] = "Honda"
        row["AccidentArea"] = "Urban"
        row["Sex"] = "Male" if i % 2 == 0 else "Female"
        row[MOTOR_TARGET] = 1 if i % 5 == 0 else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _tiny_health_tables(n: int = 10) -> HealthRawTables:
    providers = [f"PRV{i}" for i in range(n)]
    main = pd.DataFrame(
        {"Provider": providers, "PotentialFraud": ["Yes" if i % 3 == 0 else "No" for i in range(n)]}
    )
    bene_rows, ip_rows, op_rows = [], [], []
    for i, prov in enumerate(providers):
        bene_id = f"BENE{i}"
        bene_rows.append(
            {"BeneID": bene_id, "DOD": None, "ChronicCond_Alzheimer": 1, "RenalDiseaseIndicator": "0"}
        )
        ip_rows.append(
            {
                "ClaimID": f"CLMI{i}",
                "BeneID": bene_id,
                "Provider": prov,
                "AttendingPhysician": f"PHY{i}",
                "ClmDiagnosisCode_1": "D1",
            }
        )
        op_rows.append(
            {
                "ClaimID": f"CLMO{i}",
                "BeneID": bene_id,
                "Provider": prov,
                "AttendingPhysician": f"PHY{i}",
                "ClmDiagnosisCode_1": None,
            }
        )
    return HealthRawTables(
        main=main,
        beneficiary=pd.DataFrame(bene_rows),
        inpatient=pd.DataFrame(ip_rows),
        outpatient=pd.DataFrame(op_rows),
    )


# ---------------------------------------------------------------------------
# build_motor_dataset
# ---------------------------------------------------------------------------


def test_build_motor_dataset_applies_sample_size(monkeypatch):
    claims = _tiny_motor_claims(150)
    monkeypatch.setattr(data_loader, "load_motor_claims", lambda: claims)

    dataset = data_loader.build_motor_dataset(sample_size=100, seed=1)

    assert len(dataset.claims) == 100
    assert len(dataset.X) == 100
    assert len(dataset.y) == 100
    assert CLAIM_KEY_COL in dataset.X.columns
    assert set(dataset.X[CLAIM_KEY_COL]) == set(dataset.structured_df[CLAIM_KEY_COL])


def test_build_motor_dataset_no_sampling_when_n_exceeds_data(monkeypatch):
    claims = _tiny_motor_claims(150)
    monkeypatch.setattr(data_loader, "load_motor_claims", lambda: claims)

    dataset = data_loader.build_motor_dataset(sample_size=10_000, seed=1)

    assert len(dataset.claims) == 150


def test_build_motor_dataset_missing_raw_file_raises(monkeypatch):
    def _raise():
        raise DatasetFileMissingError("motor csv not found")

    monkeypatch.setattr(data_loader, "load_motor_claims", _raise)

    with pytest.raises(data_loader.DatasetNotAvailableError):
        data_loader.build_motor_dataset()


def test_motor_claim_row_lookup(monkeypatch):
    claims = _tiny_motor_claims(150)
    monkeypatch.setattr(data_loader, "load_motor_claims", lambda: claims)
    dataset = data_loader.build_motor_dataset(sample_size=100, seed=1)

    claim_key = dataset.X.iloc[0][CLAIM_KEY_COL]
    row = data_loader.motor_claim_row(dataset, claim_key)
    assert row["Make"] == "Honda"


def test_motor_claim_row_missing_key_raises(monkeypatch):
    claims = _tiny_motor_claims(150)
    monkeypatch.setattr(data_loader, "load_motor_claims", lambda: claims)
    dataset = data_loader.build_motor_dataset(sample_size=100, seed=1)

    with pytest.raises(data_loader.DatasetNotAvailableError):
        data_loader.motor_claim_row(dataset, "claim:99999999")


# ---------------------------------------------------------------------------
# build_health_dataset
# ---------------------------------------------------------------------------


def test_build_health_dataset_shapes(monkeypatch):
    tables = _tiny_health_tables(10)
    monkeypatch.setattr(data_loader, "load_health_train_tables", lambda: tables)

    dataset = data_loader.build_health_dataset(
        inpatient_sample_size=10, outpatient_sample_size=10, seed=1
    )

    assert len(dataset.X) == len(dataset.y) == len(dataset.claims)
    assert CLAIM_KEY_COL in dataset.X.columns


def test_build_health_dataset_missing_raw_file_raises(monkeypatch):
    def _raise():
        raise DatasetFileMissingError("health csv not found")

    monkeypatch.setattr(data_loader, "load_health_train_tables", _raise)

    with pytest.raises(data_loader.DatasetNotAvailableError):
        data_loader.build_health_dataset()


def test_health_claim_row_lookup(monkeypatch):
    tables = _tiny_health_tables(10)
    monkeypatch.setattr(data_loader, "load_health_train_tables", lambda: tables)
    dataset = data_loader.build_health_dataset(inpatient_sample_size=10, outpatient_sample_size=10, seed=1)

    claim_key = dataset.X.iloc[0][CLAIM_KEY_COL]
    row = data_loader.health_claim_row(dataset, claim_key)
    assert row["ClaimID"] == claim_key.split(":", 1)[1]


def test_health_claim_row_missing_key_raises(monkeypatch):
    tables = _tiny_health_tables(10)
    monkeypatch.setattr(data_loader, "load_health_train_tables", lambda: tables)
    dataset = data_loader.build_health_dataset(inpatient_sample_size=10, outpatient_sample_size=10, seed=1)

    with pytest.raises(data_loader.DatasetNotAvailableError):
        data_loader.health_claim_row(dataset, "claim:NOPE")


# ---------------------------------------------------------------------------
# cached wrappers (smoke: same args -> same object)
# ---------------------------------------------------------------------------


def test_get_cached_motor_dataset_avoids_reloading_raw_data(monkeypatch):
    claims = _tiny_motor_claims(150)
    calls = []

    def _load():
        calls.append(1)
        return claims

    monkeypatch.setattr(data_loader, "load_motor_claims", _load)

    a = data_loader.get_cached_motor_dataset(sample_size=100, seed=1)
    b = data_loader.get_cached_motor_dataset(sample_size=100, seed=1)
    assert len(calls) == 1  # second call served from st.cache_data, not re-loaded
    assert a.claims.equals(b.claims)
    assert a.X.equals(b.X)


def test_get_cached_health_dataset_avoids_reloading_raw_data(monkeypatch):
    tables = _tiny_health_tables(10)
    calls = []

    def _load():
        calls.append(1)
        return tables

    monkeypatch.setattr(data_loader, "load_health_train_tables", _load)

    a = data_loader.get_cached_health_dataset(inpatient_sample_size=10, outpatient_sample_size=10, seed=1)
    b = data_loader.get_cached_health_dataset(inpatient_sample_size=10, outpatient_sample_size=10, seed=1)
    assert len(calls) == 1
    assert a.claims.equals(b.claims)
    assert a.X.equals(b.X)
