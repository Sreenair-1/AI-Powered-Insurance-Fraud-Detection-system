"""Unit tests for src/pipeline/inference.py: end-to-end claim risk pipeline.

Tiny synthetic/in-memory fixtures only, no data/raw/ dependency, mirroring
the fixture style already used in tests/test_model_health.py and
tests/test_model_motor.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schemas import MOTOR_ALL_COLUMNS, MOTOR_TARGET
from src.data.splits import assign_claims_to_provider_splits
from src.data.synthetic import FraudRingConfig, assign_synthetic_entities, inject_fraud_rings
from src.explainability.health import HealthExplainabilityError
from src.features.fusion import fuse_health_claims, fuse_motor_claims
from src.features.structured import CLAIM_KEY_COL
from src.graph import construction
from src.models.evaluation import compute_classification_metrics
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel, MotorModelError
from src.pipeline.inference import (
    DEFAULT_RISK_THRESHOLDS,
    HEALTH_DOMAIN,
    MOTOR_DOMAIN,
    RISK_CATEGORY_DESCRIPTIONS,
    ClaimRiskAssessment,
    PipelineError,
    PipelineResult,
    RiskThresholds,
    assess_health_claims,
    assess_motor_claims,
    assess_single_health_claim,
    assess_single_motor_claim,
    categorize_risk,
    run_health_pipeline,
    run_motor_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_health_features(n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            CLAIM_KEY_COL: [f"claim:C{i}" for i in range(n)],
            "claim_type": pd.Series(rng.choice(["inpatient", "outpatient"], size=n)).astype("category"),
            "num_diagnosis_codes": rng.integers(0, 5, size=n),
            "num_procedure_codes": rng.integers(0, 3, size=n),
            "is_deceased": rng.choice([True, False], size=n),
            "chronic_condition_count": rng.integers(0, 6, size=n),
            "has_renal_disease": rng.choice([True, False], size=n),
            "degree_centrality": rng.random(n),
            "has_shared_entity": rng.choice([True, False], size=n),
        }
    )
    y = pd.Series((rng.random(n) < 0.15).astype(int), name="target")
    if y.sum() == 0:
        y.iloc[0] = 1
    if (y == 0).sum() == 0:
        y.iloc[1] = 0
    return X, y


def _fitted_health_model(n: int = 40, seed: int = 0):
    X, y = _synthetic_health_features(n=n, seed=seed)
    model = HealthFraudModel().fit(X, y)
    return model, X, y


def _synthetic_motor_features(n: int = 40, seed: int = 0):
    rng = np.random.RandomState(seed)
    numeric = rng.normal(size=n)
    categories = pd.Categorical(rng.choice(["A", "B", "C"], size=n))
    y_values = (numeric > 1.2).astype(int)
    if y_values.sum() == 0:
        y_values[0] = 1
    if (y_values == 0).sum() == 0:
        y_values[1] = 0
    X = pd.DataFrame(
        {
            CLAIM_KEY_COL: [f"claim:{i}" for i in range(n)],
            "numeric_feature": numeric,
            "cat_feature": categories,
        }
    )
    y = pd.Series(y_values, name="target")
    return X, y


def _fitted_motor_model(n: int = 40, seed: int = 0):
    X, y = _synthetic_motor_features(n=n, seed=seed)
    model = MotorFraudModel().fit(X, y)
    return model, X, y


def _augmented_motor_claims_df(n: int = 30):
    claims = pd.DataFrame({c: 0 for c in MOTOR_ALL_COLUMNS}, index=range(n))
    claims["Make"] = "Honda"
    claims[MOTOR_TARGET] = 0
    with_entities = assign_synthetic_entities(claims, seed=42, garage_pool_ratio=0.1, bank_pool_ratio=0.1)
    augmented, _ = inject_fraud_rings(
        with_entities, config=FraudRingConfig(num_rings=1, ring_size=3, seed=42)
    )
    augmented.loc[augmented.index[:4], MOTOR_TARGET] = 1
    return augmented


def _tiny_provider_grouped_health_data():
    providers = [f"PRV{i}" for i in range(1, 9)]
    labels = ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
    provider_labels = pd.DataFrame({"Provider": providers, "PotentialFraud": labels})
    provider_split_map = {
        "PRV1": "train",
        "PRV2": "train",
        "PRV3": "train",
        "PRV4": "train",
        "PRV5": "val",
        "PRV6": "val",
        "PRV7": "test",
        "PRV8": "test",
    }
    ip_rows, op_rows, bene_rows = [], [], []
    for i, prov in enumerate(providers):
        bene_id = f"BENE{i}"
        bene_rows.append(
            {
                "BeneID": bene_id,
                "DOD": None,
                "ChronicCond_Alzheimer": 1 if i % 2 == 0 else 2,
                "RenalDiseaseIndicator": "Y" if i % 3 == 0 else "0",
            }
        )
        ip_rows.append(
            {
                "ClaimID": f"CLMI{i}",
                "BeneID": bene_id,
                "Provider": prov,
                "AttendingPhysician": f"PHY{i}",
                "ClmDiagnosisCode_1": "D1",
                "ClmDiagnosisCode_2": None,
            }
        )
        op_rows.append(
            {
                "ClaimID": f"CLMO{i}",
                "BeneID": bene_id,
                "Provider": prov,
                "AttendingPhysician": f"PHY{i}",
                "ClmDiagnosisCode_1": None,
                "ClmDiagnosisCode_2": None,
            }
        )
    inpatient = pd.DataFrame(ip_rows)
    outpatient = pd.DataFrame(op_rows)
    beneficiary = pd.DataFrame(bene_rows)
    return inpatient, outpatient, beneficiary, provider_labels, provider_split_map


def _fuse_health_split(inpatient, outpatient, beneficiary, provider_labels, provider_split_map, split_name):
    ip_split = assign_claims_to_provider_splits(inpatient, "Provider", provider_split_map)
    op_split = assign_claims_to_provider_splits(outpatient, "Provider", provider_split_map)
    ip = inpatient[ip_split == split_name].reset_index(drop=True)
    op = outpatient[op_split == split_name].reset_index(drop=True)
    bene_ids = set(ip["BeneID"]) | set(op["BeneID"])
    bene = beneficiary[beneficiary["BeneID"].isin(bene_ids)].reset_index(drop=True)
    graph = construction.build_health_graph(pd.concat([ip, op], ignore_index=True, sort=False))
    return fuse_health_claims(ip, op, bene, provider_labels, graph)


# ---------------------------------------------------------------------------
# categorize_risk / RiskThresholds
# ---------------------------------------------------------------------------


def test_categorize_risk_boundaries():
    assert categorize_risk(0.0) == "low"
    assert categorize_risk(0.29) == "low"
    assert categorize_risk(0.3) == "medium"
    assert categorize_risk(0.69) == "medium"
    assert categorize_risk(0.7) == "high"
    assert categorize_risk(1.0) == "high"


def test_categorize_risk_rejects_out_of_range_probability():
    with pytest.raises(PipelineError):
        categorize_risk(-0.01)
    with pytest.raises(PipelineError):
        categorize_risk(1.01)


def test_categorize_risk_respects_custom_thresholds():
    thresholds = RiskThresholds(medium=0.1, high=0.2)
    assert categorize_risk(0.15, thresholds) == "medium"
    assert categorize_risk(0.25, thresholds) == "high"


def test_risk_thresholds_rejects_invalid_bounds():
    with pytest.raises(PipelineError):
        RiskThresholds(medium=-0.1, high=0.5)
    with pytest.raises(PipelineError):
        RiskThresholds(medium=0.5, high=1.5)
    with pytest.raises(PipelineError):
        RiskThresholds(medium=0.7, high=0.3)
    with pytest.raises(PipelineError):
        RiskThresholds(medium=0.5, high=0.5)


def test_risk_category_descriptions_cover_every_category():
    assert set(RISK_CATEGORY_DESCRIPTIONS) == {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# assess_health_claims
# ---------------------------------------------------------------------------


def test_assess_health_claims_basic_shape():
    model, X, _ = _fitted_health_model()
    assessments = assess_health_claims(model, X)

    assert len(assessments) == len(X)
    expected_proba = model.predict_proba(X)
    for i, a in enumerate(assessments):
        assert isinstance(a, ClaimRiskAssessment)
        assert a.domain == HEALTH_DOMAIN
        assert a.claim_key == X.iloc[i][CLAIM_KEY_COL]
        assert a.fraud_probability == pytest.approx(float(expected_proba[i]))
        assert a.risk_category == categorize_risk(a.fraud_probability)
        assert 1 <= len(a.top_factors) <= 10


def test_assess_health_claims_top_factors_sorted_by_abs_shap_descending():
    model, X, _ = _fitted_health_model()
    assessments = assess_health_claims(model, X, top_n=5)
    for a in assessments:
        abs_values = [abs(f["shap_value"]) for f in a.top_factors]
        assert abs_values == sorted(abs_values, reverse=True)
        assert len(a.top_factors) <= 5


def test_assess_health_claims_feature_value_matches_input_row():
    model, X, _ = _fitted_health_model()
    assessments = assess_health_claims(model, X)
    for i, a in enumerate(assessments):
        for factor in a.top_factors:
            assert factor["feature_value"] == X.iloc[i][factor["feature"]]


def test_assess_health_claims_requires_fitted_model():
    unfit_model = HealthFraudModel()
    _, X, _ = _fitted_health_model()
    with pytest.raises(HealthExplainabilityError):
        assess_health_claims(unfit_model, X)


def test_assess_health_claims_deterministic():
    model, X, _ = _fitted_health_model()
    first = assess_health_claims(model, X)
    second = assess_health_claims(model, X)
    for a1, a2 in zip(first, second):
        assert a1.fraud_probability == pytest.approx(a2.fraud_probability)
        assert a1.top_factors == a2.top_factors


# ---------------------------------------------------------------------------
# assess_motor_claims
# ---------------------------------------------------------------------------


def test_assess_motor_claims_basic_shape():
    model, X, _ = _fitted_motor_model()
    assessments = assess_motor_claims(model, X)

    assert len(assessments) == len(X)
    expected_proba = model.predict_proba(X)
    for i, a in enumerate(assessments):
        assert a.domain == MOTOR_DOMAIN
        assert a.claim_key == X.iloc[i][CLAIM_KEY_COL]
        assert a.fraud_probability == pytest.approx(float(expected_proba[i]))
        assert a.risk_category == categorize_risk(a.fraud_probability)
        assert 1 <= len(a.top_factors) <= 10


def test_assess_motor_claims_requires_fitted_model():
    unfit_model = MotorFraudModel()
    _, X, _ = _fitted_motor_model()
    with pytest.raises(MotorModelError):
        assess_motor_claims(unfit_model, X)


def test_assess_motor_claims_feature_value_matches_input_row():
    model, X, _ = _fitted_motor_model()
    assessments = assess_motor_claims(model, X)
    for i, a in enumerate(assessments):
        for factor in a.top_factors:
            assert factor["feature_value"] == X.iloc[i][factor["feature"]]


# ---------------------------------------------------------------------------
# Single-claim convenience wrappers
# ---------------------------------------------------------------------------


def test_assess_single_health_claim_matches_batch_result():
    model, X, _ = _fitted_health_model()
    batch = assess_health_claims(model, X)
    single = assess_single_health_claim(model, X.iloc[[0]])
    assert single.claim_key == batch[0].claim_key
    assert single.fraud_probability == pytest.approx(batch[0].fraud_probability)


def test_assess_single_health_claim_rejects_multi_row_input():
    model, X, _ = _fitted_health_model()
    with pytest.raises(PipelineError):
        assess_single_health_claim(model, X.iloc[:2])


def test_assess_single_motor_claim_matches_batch_result():
    model, X, _ = _fitted_motor_model()
    batch = assess_motor_claims(model, X)
    single = assess_single_motor_claim(model, X.iloc[[0]])
    assert single.claim_key == batch[0].claim_key
    assert single.fraud_probability == pytest.approx(batch[0].fraud_probability)


def test_assess_single_motor_claim_rejects_multi_row_input():
    model, X, _ = _fitted_motor_model()
    with pytest.raises(PipelineError):
        assess_single_motor_claim(model, X.iloc[:2])


# ---------------------------------------------------------------------------
# ClaimRiskAssessment.to_ui_summary
# ---------------------------------------------------------------------------


def test_claim_risk_assessment_to_ui_summary_is_plain_json_friendly():
    model, X, _ = _fitted_health_model()
    a = assess_health_claims(model, X.iloc[[0]])[0]
    summary = a.to_ui_summary()

    assert summary["claim_key"] == a.claim_key
    assert summary["domain"] == HEALTH_DOMAIN
    assert summary["risk_category_description"] == RISK_CATEGORY_DESCRIPTIONS[a.risk_category]
    assert isinstance(summary["fraud_probability"], float)
    assert isinstance(summary["top_factors"], list)
    for factor in summary["top_factors"]:
        assert isinstance(factor, dict)
        assert not isinstance(factor["shap_value"], np.generic)


# ---------------------------------------------------------------------------
# run_health_pipeline / run_motor_pipeline
# ---------------------------------------------------------------------------


def test_run_health_pipeline_without_labels_has_no_metrics():
    model, X, _ = _fitted_health_model()
    result = run_health_pipeline(model, X)
    assert isinstance(result, PipelineResult)
    assert result.domain == HEALTH_DOMAIN
    assert result.metrics is None
    assert len(result.assessments) == len(X)


def test_run_health_pipeline_with_labels_matches_evaluate_model():
    model, X, y = _fitted_health_model()
    result = run_health_pipeline(model, X, y_true=y)
    expected_metrics = compute_classification_metrics(y, model.predict_proba(X))
    assert result.metrics is not None
    assert result.metrics.accuracy == pytest.approx(expected_metrics.accuracy)
    assert result.metrics.roc_auc == pytest.approx(expected_metrics.roc_auc)


def test_run_motor_pipeline_without_labels_has_no_metrics():
    model, X, _ = _fitted_motor_model()
    result = run_motor_pipeline(model, X)
    assert result.metrics is None
    assert len(result.assessments) == len(X)


def test_run_motor_pipeline_with_labels_matches_evaluate_model():
    model, X, y = _fitted_motor_model()
    result = run_motor_pipeline(model, X, y_true=y)
    expected_metrics = compute_classification_metrics(y, model.predict_proba(X))
    assert result.metrics is not None
    assert result.metrics.precision == pytest.approx(expected_metrics.precision)
    assert result.metrics.f1 == pytest.approx(expected_metrics.f1)


def test_pipeline_result_to_ui_summary_shape():
    model, X, y = _fitted_motor_model()
    result = run_motor_pipeline(model, X, y_true=y)
    summary = result.to_ui_summary()
    assert summary["domain"] == MOTOR_DOMAIN
    assert len(summary["assessments"]) == len(X)
    assert set(summary["metrics"]) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}


# ---------------------------------------------------------------------------
# End-to-end integration with fuse_health_claims() / fuse_motor_claims()
# ---------------------------------------------------------------------------


def test_end_to_end_health_pipeline_through_fusion():
    inpatient, outpatient, beneficiary, provider_labels, provider_split_map = (
        _tiny_provider_grouped_health_data()
    )
    X_train, y_train = _fuse_health_split(
        inpatient, outpatient, beneficiary, provider_labels, provider_split_map, "train"
    )
    X_val, y_val = _fuse_health_split(
        inpatient, outpatient, beneficiary, provider_labels, provider_split_map, "val"
    )

    model = HealthFraudModel().fit(X_train, y_train)
    result = run_health_pipeline(model, X_val, y_true=y_val)

    assert len(result.assessments) == len(X_val)
    assert result.metrics is not None
    assert result.metrics.n_samples == len(X_val)
    for a in result.assessments:
        assert a.risk_category in {"low", "medium", "high"}


def test_end_to_end_motor_pipeline_through_fusion():
    claims = _augmented_motor_claims_df()
    graph = construction.build_motor_graph(claims)
    X, y = fuse_motor_claims(claims, graph)

    model = MotorFraudModel().fit(X, y)
    result = run_motor_pipeline(model, X, y_true=y)

    assert len(result.assessments) == len(X)
    assert result.metrics is not None
    assert MOTOR_TARGET not in {f["feature"] for a in result.assessments for f in a.top_factors}
