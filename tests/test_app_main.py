"""AppTest-based integration test for app/main.py.

Runs the actual Streamlit script end-to-end via `streamlit.testing.v1.AppTest`
(no browser, no data/raw/ dependency): persisted models are tiny fixtures
saved to `tmp_path` and pointed at via `FRAUD_APP_*_MODEL_DIR` env vars, and
the raw data loaders `app.data_loader` calls are monkeypatched to tiny
in-memory fixtures -- same fixture style as tests/test_app_data_loader.py
and tests/test_app_claim_view.py.

Does not exercise the optional document-upload/forensic-rescoring panel
(AppTest has no first-class file_uploader interaction support) -- that flow
is covered directly in tests/test_app_claim_view.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import data_loader
from app.config import HEALTH_MODEL_DIR_ENV, MOTOR_MODEL_DIR_ENV
from src.data.loaders import HealthRawTables
from src.data.schemas import MOTOR_ALL_COLUMNS, MOTOR_TARGET
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.models.persistence import save_health_model, save_motor_model

APP_MAIN_PATH = Path(__file__).resolve().parents[1] / "app" / "main.py"


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


@pytest.fixture
def app_environment(tmp_path, monkeypatch):
    """Point the app at tiny fixture data + freshly-fit tiny models."""
    motor_claims = _tiny_motor_claims(150)
    health_tables = _tiny_health_tables(10)
    monkeypatch.setattr(data_loader, "load_motor_claims", lambda: motor_claims)
    monkeypatch.setattr(data_loader, "load_health_train_tables", lambda: health_tables)

    motor_dataset = data_loader.build_motor_dataset(sample_size=100, seed=1)
    motor_model = MotorFraudModel().fit(motor_dataset.X, motor_dataset.y)
    motor_dir = save_motor_model(motor_model, tmp_path / "motor_model")

    health_dataset = data_loader.build_health_dataset(
        inpatient_sample_size=10, outpatient_sample_size=10, seed=1
    )
    health_model = HealthFraudModel().fit(health_dataset.X, health_dataset.y)
    health_dir = save_health_model(health_model, tmp_path / "health_model")

    monkeypatch.setenv(MOTOR_MODEL_DIR_ENV, str(motor_dir))
    monkeypatch.setenv(HEALTH_MODEL_DIR_ENV, str(health_dir))
    return motor_dataset, health_dataset


def _markdown_text(at: AppTest) -> str:
    return " ".join(str(m.value) for m in at.markdown)


def test_app_loads_overview_by_default(app_environment):
    at = AppTest.from_file(str(APP_MAIN_PATH), default_timeout=60)
    at.run()

    assert not at.exception
    text = _markdown_text(at)
    assert "InsureSight AI" in text
    assert "Investigation overview" in text
    assert len(at.radio) >= 2


def test_app_claims_investigation_and_inspect(app_environment):
    at = AppTest.from_file(str(APP_MAIN_PATH), default_timeout=60)
    at.run()
    assert not at.exception

    at.radio[0].set_value("Claims Investigation")
    at.run()
    assert not at.exception
    assert "Claims workspace" in _markdown_text(at)
    inspect = [b for b in at.button if "Inspect claim" in str(b.label)]
    assert inspect
    inspect[0].click()
    at.run()
    assert not at.exception
    assert any("fraud probability" in m.label.lower() or "model fraud probability" in m.label.lower() for m in at.metric)


def test_app_switches_to_health_domain(app_environment):
    at = AppTest.from_file(str(APP_MAIN_PATH), default_timeout=60)
    at.run()
    assert not at.exception

    at.radio[1].set_value("health")
    at.run()

    assert not at.exception
    assert "Investigation overview" in _markdown_text(at)


def test_app_model_insights_page(app_environment):
    at = AppTest.from_file(str(APP_MAIN_PATH), default_timeout=60)
    at.run()
    at.radio[0].set_value("Model Insights")
    at.run()
    assert not at.exception
    assert "Transparency and evaluation" in _markdown_text(at)


def test_app_document_forensics_page(app_environment):
    at = AppTest.from_file(str(APP_MAIN_PATH), default_timeout=60)
    at.run()
    at.radio[0].set_value("Document Forensics")
    at.run()
    assert not at.exception
    assert "Document workspace" in _markdown_text(at)


def test_app_shows_actionable_error_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv(MOTOR_MODEL_DIR_ENV, str(tmp_path / "no_such_model"))
    monkeypatch.setenv(HEALTH_MODEL_DIR_ENV, str(tmp_path / "no_such_model_either"))

    at = AppTest.from_file(str(APP_MAIN_PATH), default_timeout=60)
    at.run()

    assert not at.exception  # st.error + st.stop(), not an uncaught exception
    assert any("train_models.py" in e.value for e in at.error)
