"""Unit tests for app/claim_view.py.

Builds a tiny end-to-end dataset via app.data_loader (monkeypatched raw
loaders, same fixture style as tests/test_app_data_loader.py), fits a tiny
model directly on its fused output, then exercises the claim-assessment and
forensic-rescoring flows exactly as app/main.py would call them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app import claim_view, data_loader
from src.data.loaders import HealthRawTables
from src.data.schemas import MOTOR_ALL_COLUMNS, MOTOR_TARGET
from src.features.structured import CLAIM_KEY_COL
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.pipeline.inference import ClaimRiskAssessment


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


def _make_image(size: tuple[int, int] = (64, 64)) -> Image.Image:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def motor_dataset_and_model(monkeypatch):
    claims = _tiny_motor_claims(150)
    monkeypatch.setattr(data_loader, "load_motor_claims", lambda: claims)
    dataset = data_loader.build_motor_dataset(sample_size=100, seed=1)
    model = MotorFraudModel().fit(dataset.X, dataset.y)
    return dataset, model


@pytest.fixture
def health_dataset_and_model(monkeypatch):
    tables = _tiny_health_tables(10)
    monkeypatch.setattr(data_loader, "load_health_train_tables", lambda: tables)
    dataset = data_loader.build_health_dataset(inpatient_sample_size=10, outpatient_sample_size=10, seed=1)
    model = HealthFraudModel().fit(dataset.X, dataset.y)
    return dataset, model


# ---------------------------------------------------------------------------
# assess_claim / claim_row
# ---------------------------------------------------------------------------


def test_assess_claim_motor(motor_dataset_and_model):
    dataset, model = motor_dataset_and_model
    claim_key = dataset.X.iloc[0][CLAIM_KEY_COL]
    row = claim_view.claim_row(dataset, claim_key)

    assessment = claim_view.assess_claim(claim_view.MOTOR_DOMAIN, model, row)

    assert isinstance(assessment, ClaimRiskAssessment)
    assert assessment.claim_key == claim_key
    assert 0.0 <= assessment.fraud_probability <= 1.0
    assert assessment.risk_category in {"low", "medium", "high"}


def test_assess_claim_health(health_dataset_and_model):
    dataset, model = health_dataset_and_model
    claim_key = dataset.X.iloc[0][CLAIM_KEY_COL]
    row = claim_view.claim_row(dataset, claim_key)

    assessment = claim_view.assess_claim(claim_view.HEALTH_DOMAIN, model, row)

    assert isinstance(assessment, ClaimRiskAssessment)
    assert assessment.claim_key == claim_key


def test_assess_claim_unknown_domain_raises(motor_dataset_and_model):
    dataset, model = motor_dataset_and_model
    row = claim_view.claim_row(dataset, dataset.X.iloc[0][CLAIM_KEY_COL])
    with pytest.raises(claim_view.ClaimViewError):
        claim_view.assess_claim("unknown", model, row)


def test_claim_row_missing_key_raises(motor_dataset_and_model):
    dataset, _ = motor_dataset_and_model
    with pytest.raises(claim_view.ClaimViewError):
        claim_view.claim_row(dataset, "claim:does-not-exist")


# ---------------------------------------------------------------------------
# run_forensic_checks
# ---------------------------------------------------------------------------


def test_run_forensic_checks_valid_image(tmp_path: Path):
    path = tmp_path / "doc.jpg"
    _make_image().save(path, format="JPEG")

    result = claim_view.run_forensic_checks(path)

    assert result.ela is not None
    assert result.copy_move is not None
    assert result.metadata is not None
    assert result.any_available


def test_run_forensic_checks_missing_file_returns_all_none(tmp_path: Path):
    result = claim_view.run_forensic_checks(tmp_path / "does_not_exist.jpg")

    assert result.ela is None
    assert result.copy_move is None
    assert result.metadata is None
    assert not result.any_available


# ---------------------------------------------------------------------------
# rescore_with_forensics
# ---------------------------------------------------------------------------


def test_rescore_with_forensics_motor(motor_dataset_and_model, tmp_path: Path):
    dataset, model = motor_dataset_and_model
    claim_key = dataset.X.iloc[0][CLAIM_KEY_COL]

    path = tmp_path / "doc.jpg"
    _make_image().save(path, format="JPEG")
    forensic = claim_view.run_forensic_checks(path)

    rescored = claim_view.rescore_with_forensics(
        claim_view.MOTOR_DOMAIN, model, dataset, claim_key, forensic
    )

    assert isinstance(rescored, ClaimRiskAssessment)
    assert rescored.claim_key == claim_key
    assert 0.0 <= rescored.fraud_probability <= 1.0


def test_rescore_with_forensics_health(health_dataset_and_model, tmp_path: Path):
    dataset, model = health_dataset_and_model
    claim_key = dataset.X.iloc[0][CLAIM_KEY_COL]

    path = tmp_path / "doc.jpg"
    _make_image().save(path, format="JPEG")
    forensic = claim_view.run_forensic_checks(path)

    rescored = claim_view.rescore_with_forensics(
        claim_view.HEALTH_DOMAIN, model, dataset, claim_key, forensic
    )

    assert isinstance(rescored, ClaimRiskAssessment)


def _make_pdf(num_pages: int = 1, creator: str = "Test Hospital") -> bytes:
    import io
    import pypdf
    writer = pypdf.PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Creator": creator, "/Producer": "ReportGenerator"})
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_run_forensic_checks_valid_pdf():
    pdf_bytes = _make_pdf(num_pages=2, creator="Adobe Photoshop")
    result = claim_view.run_forensic_checks(pdf_bytes, page_index=1)

    assert result.ela is not None
    assert result.copy_move is not None
    assert result.metadata is not None
    assert result.any_available
    assert result.metadata.features.software_editing_indicator is True


def test_rescore_with_forensics_pdf_motor(motor_dataset_and_model):
    dataset, model = motor_dataset_and_model
    claim_key = dataset.X.iloc[0][CLAIM_KEY_COL]

    pdf_bytes = _make_pdf(num_pages=1, creator="Canva")
    forensic = claim_view.run_forensic_checks(pdf_bytes)

    rescored = claim_view.rescore_with_forensics(
        claim_view.MOTOR_DOMAIN, model, dataset, claim_key, forensic
    )

    assert isinstance(rescored, ClaimRiskAssessment)
    assert rescored.claim_key == claim_key
    assert 0.0 <= rescored.fraud_probability <= 1.0


def test_rescore_with_forensics_pdf_health(health_dataset_and_model):
    dataset, model = health_dataset_and_model
    claim_key = dataset.X.iloc[0][CLAIM_KEY_COL]

    pdf_bytes = _make_pdf(num_pages=1)
    forensic = claim_view.run_forensic_checks(pdf_bytes)

    rescored = claim_view.rescore_with_forensics(
        claim_view.HEALTH_DOMAIN, model, dataset, claim_key, forensic
    )

    assert isinstance(rescored, ClaimRiskAssessment)


def test_rescore_with_forensics_unknown_claim_key_raises(motor_dataset_and_model, tmp_path: Path):
    dataset, model = motor_dataset_and_model
    path = tmp_path / "doc.jpg"
    _make_image().save(path, format="JPEG")
    forensic = claim_view.run_forensic_checks(path)

    with pytest.raises(claim_view.ClaimViewError):
        claim_view.rescore_with_forensics(
            claim_view.MOTOR_DOMAIN, model, dataset, "claim:does-not-exist", forensic
        )


# ---------------------------------------------------------------------------
# Standalone New Claim Creation & Assessment
# ---------------------------------------------------------------------------


def test_generate_new_claim_id():
    cid1 = claim_view.generate_new_claim_id("NEW")
    cid2 = claim_view.generate_new_claim_id("NEW")
    assert cid1.startswith("claim:NEW-")
    assert cid2.startswith("claim:NEW-")
    assert cid1 != cid2


def test_assess_new_claim_document_motor_pdf(motor_dataset_and_model):
    _, model = motor_dataset_and_model
    pdf_bytes = _make_pdf(num_pages=1, creator="Canva")

    claim_id, forensic, verdict, assessment = claim_view.assess_new_claim_document(
        domain=claim_view.MOTOR_DOMAIN,
        model=model,
        document=pdf_bytes,
    )

    assert claim_id.startswith("claim:NEW-")
    assert forensic.any_available
    assert verdict.is_tampered  # Canva is an editing software
    assert verdict.status in ("suspicious", "tampered")
    assert isinstance(assessment, ClaimRiskAssessment)
    assert assessment.claim_key == claim_id
    assert 0.0 <= assessment.fraud_probability <= 1.0


def test_assess_new_claim_document_health_image(health_dataset_and_model, tmp_path: Path):
    _, model = health_dataset_and_model
    path = tmp_path / "invoice.jpg"
    _make_image().save(path, format="JPEG")

    claim_id, forensic, verdict, assessment = claim_view.assess_new_claim_document(
        domain=claim_view.HEALTH_DOMAIN,
        model=model,
        document=path,
    )

    assert claim_id.startswith("claim:NEW-")
    assert forensic.any_available
    assert isinstance(verdict, claim_view.DocumentTamperingVerdict)
    assert isinstance(assessment, ClaimRiskAssessment)
    assert assessment.claim_key == claim_id


def test_assess_new_claim_document_image_stream(motor_dataset_and_model):
    import io
    _, model = motor_dataset_and_model
    buf = io.BytesIO()
    _make_image().save(buf, format="JPEG")
    buf.seek(0)

    claim_id, forensic, verdict, assessment = claim_view.assess_new_claim_document(
        domain=claim_view.MOTOR_DOMAIN,
        model=model,
        document=buf,
    )

    assert claim_id.startswith("claim:NEW-")
    assert forensic.ela is not None
    assert forensic.copy_move is not None
    assert forensic.metadata is not None
    assert isinstance(verdict, claim_view.DocumentTamperingVerdict)
    assert isinstance(assessment, ClaimRiskAssessment)
