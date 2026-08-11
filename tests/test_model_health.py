"""Unit tests for src/models/health.py: health XGBoost fraud risk model.

Tiny synthetic/in-memory fixtures only, no data/raw/ dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splits import assign_claims_to_provider_splits
from src.features.fusion import fuse_health_claims
from src.features.structured import CLAIM_KEY_COL
from src.graph import construction
from src.models.health import HealthFraudModel, compute_scale_pos_weight


# ---------------------------------------------------------------------------
# Plain synthetic feature-matrix fixtures — exercise the XGBoost wrapper
# mechanics directly, independent of fusion.py.
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
    return X, y


# ---------------------------------------------------------------------------
# compute_scale_pos_weight
# ---------------------------------------------------------------------------


def test_compute_scale_pos_weight_basic():
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert compute_scale_pos_weight(y) == pytest.approx(9.0)


def test_compute_scale_pos_weight_raises_without_positives():
    y = pd.Series([0, 0, 0])
    with pytest.raises(ValueError):
        compute_scale_pos_weight(y)


# ---------------------------------------------------------------------------
# HealthFraudModel — core train/predict/predict_proba mechanics
# ---------------------------------------------------------------------------


def test_fit_predict_predict_proba_shapes():
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    assert preds.shape == (len(X),)
    assert proba.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})
    assert ((proba >= 0) & (proba <= 1)).all()


def test_claim_key_excluded_from_fitted_features():
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    assert CLAIM_KEY_COL not in model.feature_names_


def test_default_scale_pos_weight_matches_training_class_ratio():
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    expected = compute_scale_pos_weight(y)
    assert model.model.get_params()["scale_pos_weight"] == pytest.approx(expected)


def test_explicit_scale_pos_weight_overrides_computed_value():
    X, y = _synthetic_health_features()
    model = HealthFraudModel(scale_pos_weight=3.0).fit(X, y)
    assert model.model.get_params()["scale_pos_weight"] == pytest.approx(3.0)


def test_predict_before_fit_raises():
    X, _ = _synthetic_health_features()
    model = HealthFraudModel()
    with pytest.raises(RuntimeError):
        model.predict(X)
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_predict_missing_feature_column_raises():
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    X_missing = X.drop(columns=["has_shared_entity"])
    with pytest.raises(ValueError):
        model.predict(X_missing)


def test_deterministic_fit_predict():
    X, y = _synthetic_health_features()
    model1 = HealthFraudModel().fit(X, y)
    model2 = HealthFraudModel().fit(X, y)
    np.testing.assert_array_equal(model1.predict(X), model2.predict(X))
    np.testing.assert_allclose(model1.predict_proba(X), model2.predict_proba(X))


def test_predict_on_holdout_rows_after_fit_on_train_only():
    X, y = _synthetic_health_features(n=40, seed=1)
    X_train, y_train = X.iloc[:28].reset_index(drop=True), y.iloc[:28].reset_index(drop=True)
    X_val = X.iloc[28:].reset_index(drop=True)
    model = HealthFraudModel().fit(X_train, y_train)
    proba = model.predict_proba(X_val)
    assert proba.shape == (len(X_val),)
    assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------------------
# Integration with fuse_health_claims() + the provider-grouped split
# ---------------------------------------------------------------------------


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


def _fuse_split(inpatient, outpatient, beneficiary, provider_labels, provider_split_map, split_name):
    ip_split = assign_claims_to_provider_splits(inpatient, "Provider", provider_split_map)
    op_split = assign_claims_to_provider_splits(outpatient, "Provider", provider_split_map)
    ip = inpatient[ip_split == split_name].reset_index(drop=True)
    op = outpatient[op_split == split_name].reset_index(drop=True)
    bene_ids = set(ip["BeneID"]) | set(op["BeneID"])
    bene = beneficiary[beneficiary["BeneID"].isin(bene_ids)].reset_index(drop=True)
    graph = construction.build_health_graph(pd.concat([ip, op], ignore_index=True, sort=False))
    return fuse_health_claims(ip, op, bene, provider_labels, graph)


def test_health_model_respects_provider_grouped_split_end_to_end():
    inpatient, outpatient, beneficiary, provider_labels, provider_split_map = (
        _tiny_provider_grouped_health_data()
    )
    X_train, y_train = _fuse_split(
        inpatient, outpatient, beneficiary, provider_labels, provider_split_map, "train"
    )
    X_val, _ = _fuse_split(inpatient, outpatient, beneficiary, provider_labels, provider_split_map, "val")
    X_test, _ = _fuse_split(inpatient, outpatient, beneficiary, provider_labels, provider_split_map, "test")

    # sanity check: the split fixture itself has no provider spanning splits
    train_providers = {p for p, s in provider_split_map.items() if s == "train"}
    val_providers = {p for p, s in provider_split_map.items() if s == "val"}
    test_providers = {p for p, s in provider_split_map.items() if s == "test"}
    assert train_providers.isdisjoint(val_providers)
    assert train_providers.isdisjoint(test_providers)

    model = HealthFraudModel().fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test)

    assert val_proba.shape == (len(X_val),)
    assert test_proba.shape == (len(X_test),)
    assert ((val_proba >= 0) & (val_proba <= 1)).all()
    assert ((test_proba >= 0) & (test_proba <= 1)).all()


def test_fuse_health_claims_target_never_becomes_a_feature():
    inpatient, outpatient, beneficiary, provider_labels, provider_split_map = (
        _tiny_provider_grouped_health_data()
    )
    X_train, y_train = _fuse_split(
        inpatient, outpatient, beneficiary, provider_labels, provider_split_map, "train"
    )
    assert "claim_fraud_label_provider_derived" not in X_train.columns
    assert "target" not in X_train.columns
    model = HealthFraudModel().fit(X_train, y_train)
    assert "claim_fraud_label_provider_derived" not in model.feature_names_
    assert "target" not in model.feature_names_
