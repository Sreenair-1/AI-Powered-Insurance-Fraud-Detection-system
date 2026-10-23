"""Endpoint tests for the FastAPI layer in `api/`.

Same fixture style as tests/test_app_main.py: the raw-data loaders are
monkeypatched to tiny in-memory tables, and tiny models are fit on them and
saved to `tmp_path` (pointed at via `FRAUD_APP_*_MODEL_DIR`). The API itself
only ever loads those saved artifacts.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.state import AppState
from app import data_loader
from app.config import HEALTH_MODEL_DIR_ENV, MOTOR_MODEL_DIR_ENV
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.models.persistence import save_health_model, save_motor_model
from tests.test_app_main import _tiny_health_tables, _tiny_motor_claims


@pytest.fixture
def model_dirs(tmp_path, monkeypatch):
    motor_claims = _tiny_motor_claims(150)
    health_tables = _tiny_health_tables(10)
    monkeypatch.setattr(data_loader, "load_motor_claims", lambda: motor_claims)
    monkeypatch.setattr(data_loader, "load_health_train_tables", lambda: health_tables)

    motor_dataset = data_loader.build_motor_dataset(sample_size=100, seed=1)
    motor_dir = save_motor_model(MotorFraudModel().fit(motor_dataset.X, motor_dataset.y), tmp_path / "motor")
    health_dataset = data_loader.build_health_dataset(inpatient_sample_size=10, outpatient_sample_size=10, seed=1)
    health_dir = save_health_model(HealthFraudModel().fit(health_dataset.X, health_dataset.y), tmp_path / "health")

    monkeypatch.setenv(MOTOR_MODEL_DIR_ENV, str(motor_dir))
    monkeypatch.setenv(HEALTH_MODEL_DIR_ENV, str(health_dir))
    return motor_dir, health_dir


@pytest.fixture
def client(model_dirs):
    app.state.insuresight = AppState.load()
    with TestClient(app) as c:
        yield c
    del app.state.insuresight


def _png_with_duplicated_patch() -> bytes:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    arr[120:170, 120:170] = arr[20:70, 20:70]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _first_claim_key(client: TestClient, domain: str) -> str:
    return client.get(f"/domains/{domain}/claims", params={"page_size": 5}).json()["items"][0]["claim_key"]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_reports_both_models_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert {d["domain"] for d in body["domains"]} == {"health", "motor"}
    assert all(d["model_loaded"] for d in body["domains"])
    assert body["risk_bands"] == {"medium": 0.3, "high": 0.7}


def test_health_degraded_when_model_missing(model_dirs, monkeypatch, tmp_path):
    monkeypatch.setenv(MOTOR_MODEL_DIR_ENV, str(tmp_path / "does_not_exist"))
    app.state.insuresight = AppState.load()
    try:
        with TestClient(app) as c:
            body = c.get("/health").json()
            assert body["status"] == "degraded"
            motor = next(d for d in body["domains"] if d["domain"] == "motor")
            assert motor["model_loaded"] is False and "train_models.py" in motor["error"]
            assert c.get("/domains/motor/overview").status_code == 503
    finally:
        del app.state.insuresight


# ---------------------------------------------------------------------------
# Overview and claims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", ["motor", "health"])
def test_overview_counts_are_consistent(client, domain):
    r = client.get(f"/domains/{domain}/overview")
    assert r.status_code == 200
    body = r.json()
    counts = body["risk_counts"]
    assert sum(counts.values()) == body["total_claims"] > 0
    assert body["review_count"] == counts["high"] + counts["medium"]
    assert body["review_rate"] == pytest.approx(body["review_count"] / body["total_claims"])
    assert sum(b["count"] for b in body["probability_histogram"]) == body["total_claims"]
    probs = [c["fraud_probability"] for c in body["top_claims"]]
    order = {"high": 0, "medium": 1, "low": 2}
    keys = [(order[c["risk_category"]], -c["fraud_probability"]) for c in body["top_claims"]]
    assert keys == sorted(keys)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_overview_matches_streamlit_scoring(client):
    """The API's scores are the same `score_loaded_sample` output Streamlit shows."""
    from app.claim_view import score_loaded_sample
    from app.config import motor_model_dir
    from app.model_loader import load_motor_model_or_raise

    dataset = data_loader.build_motor_dataset()
    expected = score_loaded_sample(load_motor_model_or_raise(motor_model_dir()), dataset.X)
    body = client.get("/domains/motor/overview").json()
    assert body["total_claims"] == len(expected)
    assert body["risk_counts"]["high"] == int((expected["risk_category"] == "high").sum())


def test_sample_bounds_are_enforced(client):
    assert client.get("/domains/motor/overview", params={"sample_size": 50}).status_code == 422
    assert client.get("/domains/motor/overview", params={"sample_size": 5000}).status_code == 422
    assert client.get("/domains/health/overview", params={"inpatient_sample_size": 10}).status_code == 422
    assert client.get("/domains/health/overview", params={"outpatient_sample_size": 3000}).status_code == 422


def test_unknown_domain_is_404(client):
    assert client.get("/domains/life/overview").status_code == 404
    assert client.get("/domains/life/insights").status_code == 404


def test_claims_pagination_filter_and_search(client):
    body = client.get("/domains/motor/claims", params={"page_size": 10, "page": 2}).json()
    assert body["page"] == 2 and len(body["items"]) == 10
    assert body["pages"] == -(-body["total"] // 10)

    asc = client.get("/domains/motor/claims", params={"order": "asc", "page_size": 100}).json()["items"]
    probs = [i["fraud_probability"] for i in asc]
    assert probs == sorted(probs)

    for risk in ("low", "medium", "high"):
        items = client.get("/domains/motor/claims", params={"risk": [risk], "page_size": 100}).json()["items"]
        assert all(i["risk_category"] == risk for i in items)

    key = asc[0]["claim_key"]
    found = client.get("/domains/motor/claims", params={"search": key}).json()["items"]
    assert key in [i["claim_key"] for i in found]
    assert client.get("/domains/motor/claims", params={"search": "zzz-no-match"}).json()["total"] == 0


def test_claims_rejects_bad_params(client):
    assert client.get("/domains/motor/claims", params={"risk": ["severe"]}).status_code == 422
    assert client.get("/domains/motor/claims", params={"page": 0}).status_code == 422
    assert client.get("/domains/motor/claims", params={"sort": "Make"}).status_code == 422


@pytest.mark.parametrize("domain", ["motor", "health"])
def test_claim_detail(client, domain):
    key = _first_claim_key(client, domain)
    r = client.get(f"/domains/{domain}/claims/{key}")
    assert r.status_code == 200
    body = r.json()
    assessment = body["assessment"]
    assert assessment["claim_key"] == key
    assert assessment["risk_category"] in {"low", "medium", "high"}
    assert 0 < len(assessment["top_factors"]) <= 10
    assert all(f["source"] == "model_shap" for f in assessment["top_factors"])
    assert "has_shared_entity" in body["graph_signals"]
    assert body["attribute_groups"]
    # Labels never leak into the investigator view.
    assert "FraudFound_P" not in body["attributes"]
    assert "PotentialFraud" not in body["attributes"]


def test_claim_detail_unknown_claim_is_404(client):
    r = client.get("/domains/motor/claims/claim:does-not-exist")
    assert r.status_code == 404
    assert r.json()["code"] == "claim_not_found"


@pytest.mark.parametrize("domain", ["motor", "health"])
def test_claim_graph(client, domain):
    key = _first_claim_key(client, domain)
    body = client.get(f"/domains/{domain}/claims/{key}/graph").json()
    ids = {n["id"] for n in body["nodes"]}
    assert key in ids
    assert sum(n["is_focus"] for n in body["nodes"]) == 1
    assert all(e["source"] in ids and e["target"] in ids for e in body["edges"])
    assert len(body["nodes"]) <= 80
    if domain == "motor":
        assert set(body["synthetic_entities"]) == {"garage", "bank_account"}
    assert client.get(f"/domains/{domain}/claims/claim:nope/graph").status_code == 404


# ---------------------------------------------------------------------------
# Document forensics
# ---------------------------------------------------------------------------


def test_document_upload_rescores_claim(client):
    key = _first_claim_key(client, "motor")
    r = client.post(
        f"/domains/motor/claims/{key}/document",
        files={"file": ("receipt.png", _png_with_duplicated_patch(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    report = body["forensics"]
    assert report["ela"] is not None and report["ela_image"].startswith("data:image/png;base64,")
    assert report["copy_move"]["potential_duplicate_regions_found"] is True
    assert len(report["copy_move_pairs"]) == report["copy_move"]["num_filtered_matches"]
    assert report["verdict"]["status"] in {"authentic", "suspicious", "tampered"}
    assert body["after"]["fraud_probability"] >= body["before"]["fraud_probability"]
    detail = client.get(f"/domains/motor/claims/{key}").json()
    assert body["before"]["fraud_probability"] == pytest.approx(detail["assessment"]["fraud_probability"])


def test_document_upload_pdf(client):
    from tests.test_app_claim_view import _make_pdf

    key = _first_claim_key(client, "health")
    r = client.post(
        f"/domains/health/claims/{key}/document",
        files={"file": ("bill.pdf", _make_pdf(num_pages=2), "application/pdf")},
        data={"page_index": "1"},
    )
    assert r.status_code == 200, r.text
    report = r.json()["forensics"]
    assert report["is_pdf"] and report["page_count"] == 2 and report["page_index"] == 1
    assert report["metadata"]["structured"]["format"] == "PDF"


def test_document_upload_errors(client):
    key = _first_claim_key(client, "motor")
    url = f"/domains/motor/claims/{key}/document"
    assert client.post(url, files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 422
    assert client.post(url, files={"file": ("broken.png", b"not an image", "image/png")}).status_code == 422
    assert client.post(url).status_code == 422
    missing = client.post(
        "/domains/motor/claims/claim:nope/document",
        files={"file": ("r.png", _png_with_duplicated_patch(), "image/png")},
    )
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# New-claim schema and assessment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", ["motor", "health"])
def test_schema_lists_model_features(client, domain):
    body = client.get(f"/domains/{domain}/schema").json()
    names = [f["name"] for f in body["fields"]]
    insights = client.get(f"/domains/{domain}/insights").json()
    assert names == insights["feature_names"]


@pytest.mark.parametrize("domain", ["motor", "health"])
def test_assess_new_claim_without_document(client, domain):
    r = client.post(f"/domains/{domain}/assess", data={"attributes": "{}"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["claim_key"].startswith("claim:NEW-")
    assert body["forensics"] is None
    assert 0.0 <= body["assessment"]["fraud_probability"] <= 1.0


def test_assess_new_claim_with_document_and_attributes(client):
    fields = {f["name"]: f for f in client.get("/domains/motor/schema").json()["fields"]}
    attrs = {"Age": 41, "AccidentArea": fields["AccidentArea"]["options"][-1]}
    r = client.post(
        "/domains/motor/assess",
        data={"attributes": json.dumps(attrs)},
        files={"file": ("estimate.png", _png_with_duplicated_patch(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["forensics"]["verdict"]["tampering_score"] > 0
    assert any(f["source"] == "document_heuristic" for f in body["assessment"]["top_factors"])


def test_assess_validation_errors(client):
    def post(attrs):
        return client.post("/domains/motor/assess", data={"attributes": attrs})

    assert post("not json").status_code == 422
    assert post("[1, 2]").status_code == 422
    assert post(json.dumps({"NotAFeature": 1})).status_code == 422
    assert post(json.dumps({"Age": "old"})).status_code == 422
    assert post(json.dumps({"Age": -3})).status_code == 422
    assert post(json.dumps({"Age": 2.5})).status_code == 422


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------


def test_insights_passes_sidecars_through_unchanged(client, model_dirs):
    motor_dir, _ = model_dirs
    metrics = {"roc_auc": 0.8502710251302179, "precision": 0.1917808219178082, "threshold": 0.5}
    (motor_dir / "eval_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (motor_dir / "roc_curve.json").write_text(json.dumps({"fpr": [0, 1], "tpr": [0, 1]}), encoding="utf-8")

    body = client.get("/domains/motor/insights").json()
    assert body["eval_metrics"] == metrics
    assert body["roc_curve"] == {"fpr": [0, 1], "tpr": [0, 1]}
    assert body["pr_curve"] is None
    assert any("synthetic" in text for text in body["limitations"])


def test_insights_health_limitations_mention_proxy_label(client):
    body = client.get("/domains/health/insights").json()
    assert any("provider-level proxy" in text for text in body["limitations"])
