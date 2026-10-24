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
from app.config import BENCHMARKS_DIR_ENV, HEALTH_MODEL_DIR_ENV, MOTOR_MODEL_DIR_ENV, REVIEWS_PATH_ENV
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
    # Never read the real models/benchmarks/ or write the real review log from tests.
    monkeypatch.setenv(BENCHMARKS_DIR_ENV, str(tmp_path / "benchmarks"))
    monkeypatch.setenv(REVIEWS_PATH_ENV, str(tmp_path / "reviews" / "reviews.jsonl"))
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
    network = body["network_risk"]
    assert network["level"] in {"candidate_ring", "linked", "none"}
    assert network["benchmark"] is None  # no benchmark files in the test benchmarks dir
    assert body["document_benchmark"] is None


def test_claim_detail_network_risk_reports_injected_ring(client):
    """The motor fixture's seeded augmentation injects rings; a flagged claim's
    detail lists its ring, the other members and the shared-pair evidence."""
    claims = client.get("/domains/motor/claims", params={"page_size": 100}).json()["items"]
    details = [client.get(f"/domains/motor/claims/{c['claim_key']}").json() for c in claims]
    flagged = [d for d in details if d["network_risk"]["level"] == "candidate_ring"]
    assert flagged, "expected at least one injected ring to be detected in the motor sample"
    risk = flagged[0]["network_risk"]
    assert risk["ring_size"] == len(risk["ring_claims"]) + 1
    assert flagged[0]["claim_key"] not in risk["ring_claims"]
    evidence = risk["evidence"][0]
    assert {e["type"] for e in evidence["entities"]} == {"garage", "bank_account"}
    assert evidence["claims"] >= 2 and evidence["p_value"] <= 1e-3
    assert evidence["expected"] < evidence["claims"]


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


def test_benchmarks_are_passed_through_and_summarised(client, tmp_path):
    body = client.get("/domains/motor/insights").json()
    assert body["tampering_benchmark"] is None and body["ring_benchmark"] is None

    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    verdict = {"detection_rate": 1.0, "false_positive_rate": 0.9}
    tampering = {"source": "synthetic_rendered_bills", "n_bills": 2,
                 "results": {"as_submitted": {"methods": {"combined_verdict": verdict}}}}
    fuzzy = {"precision": 0.2, "recall": 0.77, "ring_recovery_rate": 0.875}
    rings = {"noise_sweep": [{"noise": 0.0, "naive_any_shared_entity": {"precision": 0.02},
                              "raw": fuzzy, "normalized": fuzzy, "fuzzy": fuzzy}]}
    (bench_dir / "tampering_benchmark.json").write_text(json.dumps(tampering), encoding="utf-8")
    (bench_dir / "ring_benchmark.json").write_text(json.dumps(rings), encoding="utf-8")

    body = client.get("/domains/health/insights").json()
    assert body["tampering_benchmark"] == tampering
    assert body["ring_benchmark"] == rings

    detail = client.get(f"/domains/motor/claims/{_first_claim_key(client, 'motor')}").json()
    assert detail["document_benchmark"]["as_submitted"] == verdict
    assert detail["network_risk"]["benchmark"] == {"noise": 0.0, **fuzzy, "naive_precision": 0.02}


def test_document_upload_rejects_decompression_bomb(client, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    key = _first_claim_key(client, "motor")
    r = client.post(
        f"/domains/motor/claims/{key}/document",
        files={"file": ("huge.png", _png_with_duplicated_patch(), "image/png")},
    )
    assert r.status_code == 422
    assert r.json()["code"] == "invalid_document"


# ---------------------------------------------------------------------------
# Workload curve, investigator reviews and audit reports
# ---------------------------------------------------------------------------


def test_insights_workload_curve_is_derived_from_saved_roc(client, model_dirs):
    motor_dir, _ = model_dirs
    (motor_dir / "eval_metrics.json").write_text(json.dumps({"n_samples": 100, "n_positive": 10}), encoding="utf-8")
    roc = {"fpr": [0.0, 0.0, 0.1, 1.0], "tpr": [0.0, 0.5, 0.9, 1.0], "thresholds": [float("inf"), 0.8, 0.4, 0.0]}
    (motor_dir / "roc_curve.json").write_text(json.dumps(roc), encoding="utf-8")

    curve = client.get("/domains/motor/insights").json()["workload_curve"]
    assert curve["split"] == "validation" and curve["prevalence"] == pytest.approx(0.1)
    second = curve["points"][1]  # flag 5 of 10 positives, no negatives: 5% of claims reviewed
    assert second["review_rate"] == pytest.approx(0.05)
    assert second["recall"] == pytest.approx(0.5)
    assert second["precision"] == pytest.approx(1.0) and second["lift"] == pytest.approx(10.0)
    assert curve["points"][0]["threshold"] is None  # +inf is not valid JSON
    assert curve["points"][-1]["review_rate"] == pytest.approx(1.0)


def test_review_is_recorded_with_score_snapshot_and_history(client, model_dirs, tmp_path):
    key = _first_claim_key(client, "motor")
    assert client.get(f"/domains/motor/claims/{key}/review").json() == {"current": None, "history": []}

    first = client.post(f"/domains/motor/claims/{key}/review",
                        json={"outcome": "needs_more_info", "note": "Ask for the repair invoice", "reviewer": " Asha "})
    assert first.status_code == 200
    record = first.json()["current"]
    assert record["retraining_label"] is None and record["reviewer"] == "Asha"
    assert 0.0 <= record["fraud_probability_at_review"] <= 1.0
    assert record["risk_category_at_review"] in {"low", "medium", "high"}
    assert len(record["model_version"]) == 12

    confirm = {"outcome": "confirmed_suspicious", "note": "Invoice altered"}
    client.post(f"/domains/motor/claims/{key}/review", json=confirm)
    body = client.get(f"/domains/motor/claims/{key}/review").json()
    assert [r["outcome"] for r in body["history"]] == ["needs_more_info", "confirmed_suspicious"]
    assert body["current"]["retraining_label"] == 1

    log = (tmp_path / "reviews" / "reviews.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 2  # append-only: the first decision is kept

    summary = client.get("/domains/motor/reviews").json()
    assert summary["reviewed_claims"] == 1
    assert summary["counts"] == {"confirmed_suspicious": 1, "cleared": 0, "needs_more_info": 0}
    assert summary["labels"] == [
        {"claim_key": key, "label": 1, "recorded_at": body["current"]["recorded_at"],
         "model_version": body["current"]["model_version"]}
    ]
    assert client.get("/domains/health/reviews").json()["reviewed_claims"] == 0


def test_review_validation(client):
    url = f"/domains/motor/claims/{_first_claim_key(client, 'motor')}/review"
    assert client.post(url, json={"outcome": "fraud"}).status_code == 422
    assert client.post(url, json={"outcome": "cleared", "note": "x" * 4001}).status_code == 422
    assert client.post("/domains/motor/claims/claim:nope/review", json={"outcome": "cleared"}).status_code == 404
    assert client.get("/domains/motor/claims/claim:nope/review").status_code == 404


def test_audit_report_json_is_complete_and_verifiable(client):
    import hashlib

    key = _first_claim_key(client, "motor")
    client.post(f"/domains/motor/claims/{key}/review", json={"outcome": "cleared", "note": "Matches police report"})
    r = client.get(f"/domains/motor/claims/{key}/audit-report")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    report = r.json()
    assert report["claim_key"] == key and report["report_type"] == "claim_risk_audit"
    assert report["assessment"]["top_factors"]
    assert len(report["model"]["version"]) == 12 and report["model"]["risk_bands"] == {"medium": 0.3, "high": 0.7}
    assert report["network"]["graph_links"] and {"garage", "bank_account"} <= {
        link["entity_type"] for link in report["network"]["graph_links"]}
    assert all(link["synthetic"] for link in report["network"]["graph_links"])
    assert report["document"] is None
    assert report["review"]["current"]["outcome"] == "cleared"
    assert "FraudFound_P" not in report["attributes"]

    integrity = report.pop("integrity")
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert integrity["digest"] == hashlib.sha256(canonical).hexdigest()


def test_audit_report_with_document_and_printable_html(client):
    key = _first_claim_key(client, "motor")
    png = _png_with_duplicated_patch()
    r = client.post(f"/domains/motor/claims/{key}/audit-report",
                    files={"file": ("bill<1>.png", png, "image/png")})
    assert r.status_code == 200
    document = r.json()["document"]
    assert document["filename"] == "bill<1>.png"
    assert len(document["sha256"]) == 64
    assert document["verdict"]["status"] in {"authentic", "suspicious", "tampered"}
    assert "ela_image" not in document and "original_image" not in document

    page = client.post(f"/domains/motor/claims/{key}/audit-report", params={"format": "html"},
                       files={"file": ("bill<1>.png", png, "image/png")})
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert "Claim risk audit report" in page.text
    assert "bill&lt;1&gt;.png" in page.text and "bill<1>.png" not in page.text  # escaped
    assert client.get("/domains/motor/claims/claim:nope/audit-report").status_code == 404
