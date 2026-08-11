"""Unit tests for src/explainability/motor.py: motor SHAP explainability.

Tiny synthetic/in-memory fixtures only, no data/raw/ dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability.motor import MotorShapExplanation, explain_motor_claims
from src.features.structured import CLAIM_KEY_COL
from src.models.motor import MotorFraudModel, MotorModelError


def _synthetic_motor_features(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            CLAIM_KEY_COL: [f"claim:{i}" for i in range(n)],
            "Age": rng.integers(18, 70, size=n),
            "VehiclePrice": rng.integers(1000, 50000, size=n),
            "GarageID": rng.choice([f"GAR{i}" for i in range(5)], size=n),
            "BankAccountID": rng.choice([f"BANK{i}" for i in range(5)], size=n),
        }
    )
    y = pd.Series((rng.random(n) < 0.3).astype(int))
    if y.sum() == 0:
        y.iloc[0] = 1
    if y.sum() == n:
        y.iloc[0] = 0
    return X, y


@pytest.fixture(scope="module")
def fitted_model_and_data() -> tuple[MotorFraudModel, pd.DataFrame, pd.Series]:
    X, y = _synthetic_motor_features()
    model = MotorFraudModel().fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# explain_motor_claims — validation
# ---------------------------------------------------------------------------


def test_unfit_model_raises():
    X, _ = _synthetic_motor_features()
    model = MotorFraudModel()
    with pytest.raises(MotorModelError):
        explain_motor_claims(model, X)


def test_missing_claim_key_column_raises(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    X_no_key = X.drop(columns=[CLAIM_KEY_COL])
    with pytest.raises(MotorModelError):
        explain_motor_claims(model, X_no_key)


def test_missing_feature_column_raises(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    X_missing = X.drop(columns=["VehiclePrice"])
    with pytest.raises(MotorModelError):
        explain_motor_claims(model, X_missing)


def test_extra_feature_column_raises(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    X_extra = X.copy()
    X_extra["UnexpectedColumn"] = 1
    with pytest.raises(MotorModelError):
        explain_motor_claims(model, X_extra)


def test_never_refits_model(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    feature_columns_before = model.feature_columns
    scale_pos_weight_before = model.scale_pos_weight
    explain_motor_claims(model, X)
    assert model.feature_columns == feature_columns_before
    assert model.scale_pos_weight == pytest.approx(scale_pos_weight_before)


# ---------------------------------------------------------------------------
# explain_motor_claims — shape / structure / alignment
# ---------------------------------------------------------------------------


def test_returns_motor_shap_explanation(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    assert isinstance(explanation, MotorShapExplanation)


def test_shap_values_shape_matches_rows_and_fitted_features(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    assert explanation.shap_values.shape == (len(X), len(model.feature_columns))


def test_feature_names_match_fitted_feature_order(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    assert explanation.feature_names == model.feature_columns


def test_claim_keys_match_input_order(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    assert explanation.claim_keys == X[CLAIM_KEY_COL].astype(str).tolist()


def test_claim_key_never_appears_as_a_feature_name(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    assert CLAIM_KEY_COL not in explanation.feature_names


def test_works_on_a_row_subset_not_seen_together_at_fit_time(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    X_subset = X.iloc[5:10].reset_index(drop=True)
    explanation = explain_motor_claims(model, X_subset)
    assert explanation.shap_values.shape == (5, len(model.feature_columns))
    assert explanation.claim_keys == X_subset[CLAIM_KEY_COL].tolist()


# ---------------------------------------------------------------------------
# SHAP values reconstruct the model's own predictions (margin/log-odds space)
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def test_base_value_plus_shap_sum_reconstructs_predict_proba(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    reconstructed_margin = explanation.base_value + explanation.shap_values.sum(axis=1)
    reconstructed_proba = _sigmoid(reconstructed_margin)
    actual_proba = model.predict_proba(X)
    np.testing.assert_allclose(reconstructed_proba, actual_proba, atol=1e-4)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_across_calls(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation1 = explain_motor_claims(model, X)
    explanation2 = explain_motor_claims(model, X)
    np.testing.assert_array_equal(explanation1.shap_values, explanation2.shap_values)
    assert explanation1.base_value == pytest.approx(explanation2.base_value)


# ---------------------------------------------------------------------------
# global_importance
# ---------------------------------------------------------------------------


def test_global_importance_covers_every_feature_and_is_sorted_descending(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    importance = explanation.global_importance()
    assert set(importance.index) == set(model.feature_columns)
    assert (importance.values >= 0).all()
    assert list(importance.values) == sorted(importance.values, reverse=True)


# ---------------------------------------------------------------------------
# local_explanation
# ---------------------------------------------------------------------------


def test_local_explanation_sorted_by_absolute_value_descending(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    local = explanation.local_explanation(0)
    assert set(local.index) == set(model.feature_columns)
    abs_values = local.abs().values
    assert list(abs_values) == sorted(abs_values, reverse=True)


def test_local_explanation_values_match_shap_values_row(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    local = explanation.local_explanation(2)
    for feature, value in local.items():
        col_idx = explanation.feature_names.index(feature)
        assert value == pytest.approx(explanation.shap_values[2, col_idx])


def test_local_explanation_out_of_range_row_raises(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    with pytest.raises(MotorModelError):
        explanation.local_explanation(len(X))
    with pytest.raises(MotorModelError):
        explanation.local_explanation(-1)


# ---------------------------------------------------------------------------
# summary_records
# ---------------------------------------------------------------------------


def test_summary_records_one_per_claim_with_bounded_top_features(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    records = explanation.summary_records(top_n=2)
    assert len(records) == len(X)
    for record, claim_key in zip(records, explanation.claim_keys):
        assert record[CLAIM_KEY_COL] == claim_key
        assert record["base_value"] == pytest.approx(explanation.base_value)
        assert len(record["top_features"]) <= 2
        for feature_entry in record["top_features"]:
            assert set(feature_entry.keys()) == {"feature", "shap_value"}
            assert isinstance(feature_entry["shap_value"], float)


def test_summary_records_non_positive_top_n_raises(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    with pytest.raises(MotorModelError):
        explanation.summary_records(top_n=0)


# ---------------------------------------------------------------------------
# to_frame
# ---------------------------------------------------------------------------


def test_to_frame_shape_and_columns(fitted_model_and_data):
    model, X, _ = fitted_model_and_data
    explanation = explain_motor_claims(model, X)
    frame = explanation.to_frame()
    assert frame.shape == (len(X), len(model.feature_columns) + 1)
    assert list(frame.columns) == [CLAIM_KEY_COL] + explanation.feature_names
    assert frame[CLAIM_KEY_COL].tolist() == explanation.claim_keys
