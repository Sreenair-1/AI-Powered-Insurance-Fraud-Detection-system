"""Unit tests for src/explainability/health.py: SHAP explainability for
HealthFraudModel.

Tiny synthetic in-memory fixtures + a fitted test model only, no
data/raw/ dependency. Mirrors the fixture style of
tests/test_model_health.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schemas import HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL
from src.explainability.common import ExplainabilityError
from src.explainability.health import (
    HealthExplainabilityError,
    HealthShapExplanation,
    explain_health_claims,
    explain_single_health_claim,
)
from src.features.structured import CLAIM_KEY_COL
from src.models.health import HealthFraudModel


def _synthetic_health_features(n: int = 30, seed: int = 0):
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
    y = pd.Series((rng.random(n) < 0.3).astype(int))
    if y.sum() == 0:
        y.iloc[0] = 1
    return X, y


@pytest.fixture()
def fitted_model_and_data():
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    return model, X


# ---------------------------------------------------------------------------
# explain_health_claims — shapes, feature names, base value, determinism
# ---------------------------------------------------------------------------


def test_explain_health_claims_shapes(fitted_model_and_data):
    model, X = fitted_model_and_data
    explanation = explain_health_claims(model, X)
    assert isinstance(explanation, HealthShapExplanation)
    assert explanation.shap_values.shape == (len(X), len(model.feature_names_))
    assert explanation.feature_names == tuple(model.feature_names_)
    assert explanation.predicted_proba.shape == (len(X),)
    assert isinstance(explanation.base_value, float)
    assert explanation.n_samples == len(X)


def test_explain_health_claims_carries_claim_keys(fitted_model_and_data):
    model, X = fitted_model_and_data
    explanation = explain_health_claims(model, X)
    assert explanation.claim_keys == tuple(X[CLAIM_KEY_COL])


def test_explain_health_claims_claim_keys_none_when_absent(fitted_model_and_data):
    model, X = fitted_model_and_data
    X_no_key = X.drop(columns=[CLAIM_KEY_COL])
    explanation = explain_health_claims(model, X_no_key)
    assert explanation.claim_keys is None


def test_explain_health_claims_deterministic(fitted_model_and_data):
    model, X = fitted_model_and_data
    e1 = explain_health_claims(model, X)
    e2 = explain_health_claims(model, X)
    np.testing.assert_allclose(e1.shap_values, e2.shap_values)
    assert e1.base_value == pytest.approx(e2.base_value)


def test_explain_health_claims_additive_property_matches_raw_margin(fitted_model_and_data):
    model, X = fitted_model_and_data
    explanation = explain_health_claims(model, X)
    features = X.drop(columns=[CLAIM_KEY_COL])[model.feature_names_]
    raw_margin = model.model.predict(features, output_margin=True)
    reconstructed = explanation.base_value + explanation.shap_values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, raw_margin, atol=1e-3)


def test_explain_health_claims_preserves_fitted_feature_order(fitted_model_and_data):
    model, X = fitted_model_and_data
    # Shuffle X's columns; the explanation's feature order must still match
    # the model's fitted order, not X's incoming column order.
    shuffled_cols = [CLAIM_KEY_COL] + list(reversed([c for c in X.columns if c != CLAIM_KEY_COL]))
    explanation = explain_health_claims(model, X[shuffled_cols])
    assert explanation.feature_names == tuple(model.feature_names_)


def test_explain_health_claims_ignores_extra_label_column(fitted_model_and_data):
    model, X = fitted_model_and_data
    X_with_label = X.copy()
    X_with_label[HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL] = 1
    explanation = explain_health_claims(model, X_with_label)
    assert HEALTH_CLAIM_LABEL_PROVIDER_DERIVED_COL not in explanation.feature_names


# ---------------------------------------------------------------------------
# Reconciliation with src/explainability/common.py
# ---------------------------------------------------------------------------


def test_health_explainability_error_is_common_explainability_error():
    """health.py delegates to common.py; its error type must remain
    catchable under the shared `ExplainabilityError` base."""
    assert issubclass(HealthExplainabilityError, ExplainabilityError)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_explain_health_claims_raises_before_fit():
    unfitted = HealthFraudModel()
    X, _ = _synthetic_health_features()
    with pytest.raises(HealthExplainabilityError):
        explain_health_claims(unfitted, X)


def test_explain_health_claims_raises_on_missing_feature_column(fitted_model_and_data):
    model, X = fitted_model_and_data
    X_missing = X.drop(columns=["has_shared_entity"])
    with pytest.raises(HealthExplainabilityError):
        explain_health_claims(model, X_missing)


def test_explain_single_health_claim_raises_on_multi_row(fitted_model_and_data):
    model, X = fitted_model_and_data
    with pytest.raises(HealthExplainabilityError):
        explain_single_health_claim(model, X)


def test_claim_contributions_raises_on_out_of_range_row_index(fitted_model_and_data):
    model, X = fitted_model_and_data
    explanation = explain_health_claims(model, X)
    with pytest.raises(HealthExplainabilityError):
        explanation.claim_contributions(len(X))


# ---------------------------------------------------------------------------
# global_importance / claim_contributions / to_ui_summary
# ---------------------------------------------------------------------------


def test_global_importance_contains_all_features_sorted_desc(fitted_model_and_data):
    model, X = fitted_model_and_data
    explanation = explain_health_claims(model, X)
    global_df = explanation.global_importance()
    assert set(global_df["feature"]) == set(model.feature_names_)
    values = global_df["mean_abs_shap_value"].to_numpy()
    assert (np.diff(values) <= 1e-12).all()  # non-increasing


def test_claim_contributions_sorted_by_abs_value_desc(fitted_model_and_data):
    model, X = fitted_model_and_data
    explanation = explain_health_claims(model, X)
    contributions = explanation.claim_contributions(0)
    assert set(contributions["feature"]) == set(model.feature_names_)
    abs_values = contributions["shap_value"].abs().to_numpy()
    assert (np.diff(abs_values) <= 1e-12).all()  # non-increasing


def test_to_ui_summary_shape_and_top_n(fitted_model_and_data):
    model, X = fitted_model_and_data
    explanation = explain_health_claims(model, X)
    summary = explanation.to_ui_summary(top_n=3)
    assert summary["n_samples"] == len(X)
    assert isinstance(summary["base_value"], float)
    assert len(summary["top_global_features"]) == 3
    for entry in summary["top_global_features"]:
        assert set(entry.keys()) == {"feature", "mean_abs_shap_value"}


# ---------------------------------------------------------------------------
# explain_single_health_claim — local explanation dict
# ---------------------------------------------------------------------------


def test_explain_single_health_claim_structure(fitted_model_and_data):
    model, X = fitted_model_and_data
    row = X.iloc[[0]]
    result = explain_single_health_claim(model, row)
    assert result["claim_key"] == row.iloc[0][CLAIM_KEY_COL]
    assert 0.0 <= result["predicted_proba"] <= 1.0
    assert isinstance(result["base_value"], float)
    assert len(result["contributions"]) == len(model.feature_names_)
    for entry in result["contributions"]:
        assert set(entry.keys()) == {"feature", "feature_value", "shap_value"}
        assert entry["feature"] in model.feature_names_


def test_explain_single_health_claim_contributions_sorted_desc(fitted_model_and_data):
    model, X = fitted_model_and_data
    result = explain_single_health_claim(model, X.iloc[[1]])
    abs_values = [abs(c["shap_value"]) for c in result["contributions"]]
    assert all(abs_values[i] >= abs_values[i + 1] for i in range(len(abs_values) - 1))


def test_explain_single_health_claim_feature_values_match_input_row(fitted_model_and_data):
    model, X = fitted_model_and_data
    row = X.iloc[[2]]
    result = explain_single_health_claim(model, row)
    by_feature = {c["feature"]: c["feature_value"] for c in result["contributions"]}
    for feature in model.feature_names_:
        assert by_feature[feature] == row.iloc[0][feature]


def test_explain_single_health_claim_claim_key_none_when_absent(fitted_model_and_data):
    model, X = fitted_model_and_data
    row = X.drop(columns=[CLAIM_KEY_COL]).iloc[[0]]
    result = explain_single_health_claim(model, row)
    assert result["claim_key"] is None


# ---------------------------------------------------------------------------
# Integration with fuse_health_claims() output shape
# ---------------------------------------------------------------------------


def test_explain_health_claims_on_fuse_health_claims_style_matrix():
    """Feature matrix shaped like fuse_health_claims() output (claim_key +
    structured + graph-style columns), fit and explained end-to-end without
    going through the real fusion pipeline (keeps this test self-contained,
    tiny, and independent of data/raw/)."""
    X, y = _synthetic_health_features(n=25, seed=7)
    model = HealthFraudModel().fit(X, y)
    explanation = explain_health_claims(model, X)
    assert explanation.shap_values.shape == (25, len(model.feature_names_))
    assert explanation.claim_keys == tuple(X[CLAIM_KEY_COL])
