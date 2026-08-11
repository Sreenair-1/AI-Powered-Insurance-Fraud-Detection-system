"""Tests for the shared SHAP explainability layer (`src.explainability.common`).

Uses tiny synthetic in-memory feature matrices and a locally-fit
`xgboost.XGBClassifier` only — never `HealthFraudModel`/`MotorFraudModel`
(those are being built independently) and never any file under `data/raw/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.explainability.common import (
    ExplainabilityError,
    GlobalExplanation,
    LocalExplanation,
    create_tree_explainer,
    explain_global,
    explain_local,
    explain_one,
    prepare_features_for_shap,
)
from src.features.structured import CLAIM_KEY_COL

RANDOM_STATE = 42


def _make_features(n: int = 40) -> pd.DataFrame:
    rng = np.random.RandomState(RANDOM_STATE)
    return pd.DataFrame(
        {
            "amount": rng.uniform(100, 5000, size=n),
            "age": rng.randint(18, 80, size=n),
            "category": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        }
    )


def _fit_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=15,
        max_depth=3,
        enable_categorical=True,
        tree_method="hist",
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model


@pytest.fixture
def fitted_bundle():
    X = _make_features()
    y = pd.Series((X["amount"] > 2500).astype(int), name="target")
    model = _fit_model(X, y)
    feature_names = list(X.columns)
    return model, X, feature_names


@pytest.fixture
def fused_bundle(fitted_bundle):
    """Same fitted model, but X carries claim_key + a leakage column, like fused output."""
    model, X, feature_names = fitted_bundle
    fused = X.copy()
    fused.insert(0, CLAIM_KEY_COL, [f"claim:{i}" for i in range(len(fused))])
    fused["FraudFound_P"] = 0
    return model, fused, feature_names


# ---------------------------------------------------------------------------
# create_tree_explainer / validation
# ---------------------------------------------------------------------------


def test_create_tree_explainer_on_fitted_model(fitted_bundle):
    model, _, _ = fitted_bundle
    explainer = create_tree_explainer(model)
    assert explainer is not None


def test_create_tree_explainer_none_raises():
    with pytest.raises(ExplainabilityError, match="None"):
        create_tree_explainer(None)


def test_create_tree_explainer_unfitted_raises():
    with pytest.raises(ExplainabilityError, match="not fitted"):
        create_tree_explainer(XGBClassifier())


def test_create_tree_explainer_no_predict_proba_raises():
    class NotAModel:
        pass

    with pytest.raises(ExplainabilityError, match="predict_proba"):
        create_tree_explainer(NotAModel())


# ---------------------------------------------------------------------------
# prepare_features_for_shap
# ---------------------------------------------------------------------------


def test_prepare_features_drops_claim_key_and_leakage(fused_bundle):
    _, fused, feature_names = fused_bundle
    prepared = prepare_features_for_shap(fused, feature_names)
    assert list(prepared.columns) == feature_names
    assert CLAIM_KEY_COL not in prepared.columns
    assert "FraudFound_P" not in prepared.columns


def test_prepare_features_reorders_columns(fitted_bundle):
    _, X, feature_names = fitted_bundle
    reordered = X[list(reversed(feature_names))]
    prepared = prepare_features_for_shap(reordered, feature_names)
    assert list(prepared.columns) == feature_names


def test_prepare_features_missing_column_raises(fitted_bundle):
    _, X, feature_names = fitted_bundle
    dropped = X.drop(columns=["age"])
    with pytest.raises(ExplainabilityError, match="missing feature column"):
        prepare_features_for_shap(dropped, feature_names)


def test_prepare_features_empty_x_raises(fitted_bundle):
    _, X, feature_names = fitted_bundle
    with pytest.raises(ExplainabilityError, match="no rows"):
        prepare_features_for_shap(X.iloc[0:0], feature_names)


def test_prepare_features_non_dataframe_raises(fitted_bundle):
    _, _, feature_names = fitted_bundle
    with pytest.raises(ExplainabilityError, match="DataFrame"):
        prepare_features_for_shap([1, 2, 3], feature_names)


def test_prepare_features_empty_feature_names_raises(fitted_bundle):
    _, X, _ = fitted_bundle
    with pytest.raises(ExplainabilityError, match="feature_names is empty"):
        prepare_features_for_shap(X, [])


# ---------------------------------------------------------------------------
# explain_global
# ---------------------------------------------------------------------------


def test_explain_global_returns_all_features(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    result = explain_global(explainer, X, feature_names)

    assert isinstance(result, GlobalExplanation)
    assert set(result.feature_names) == set(feature_names)
    assert set(result.mean_abs_shap.keys()) == set(feature_names)
    assert sorted(result.ranked_feature_names) == sorted(feature_names)
    assert result.n_samples == len(X)


def test_explain_global_ranked_features_sorted_descending(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    result = explain_global(explainer, X, feature_names)

    importances = [result.mean_abs_shap[name] for name in result.ranked_feature_names]
    assert importances == sorted(importances, reverse=True)


def test_explain_global_amount_is_most_important(fitted_bundle):
    # y was derived purely from `amount`, so it must dominate global importance.
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    result = explain_global(explainer, X, feature_names)
    assert result.ranked_feature_names[0] == "amount"


def test_explain_global_top_features_respects_n(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    result = explain_global(explainer, X, feature_names)
    top = result.top_features(n=2)
    assert len(top) == 2
    assert top[0][0] == result.ranked_feature_names[0]


def test_explain_global_to_dict_has_no_dataframe_values(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    result = explain_global(explainer, X, feature_names)
    d = result.to_dict()
    assert isinstance(d["mean_abs_shap"], dict)
    assert isinstance(d["base_value"], float)


def test_explain_global_accepts_fused_style_input(fused_bundle):
    model, fused, feature_names = fused_bundle
    explainer = create_tree_explainer(model)
    result = explain_global(explainer, fused, feature_names)
    assert result.n_samples == len(fused)


# ---------------------------------------------------------------------------
# explain_local
# ---------------------------------------------------------------------------


def test_explain_local_one_result_per_row(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    results = explain_local(explainer, X, feature_names)
    assert len(results) == len(X)
    assert all(isinstance(r, LocalExplanation) for r in results)


def test_explain_local_claim_key_from_column(fused_bundle):
    model, fused, feature_names = fused_bundle
    explainer = create_tree_explainer(model)
    results = explain_local(explainer, fused, feature_names)
    assert [r.claim_key for r in results] == list(fused[CLAIM_KEY_COL])


def test_explain_local_claim_key_none_when_absent(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    results = explain_local(explainer, X, feature_names)
    assert all(r.claim_key is None for r in results)


def test_explain_local_explicit_claim_keys_override(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    keys = [f"custom:{i}" for i in range(len(X))]
    results = explain_local(explainer, X, feature_names, claim_keys=keys)
    assert [r.claim_key for r in results] == keys


def test_explain_local_claim_keys_length_mismatch_raises(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    with pytest.raises(ExplainabilityError, match="claim_keys has"):
        explain_local(explainer, X, feature_names, claim_keys=["only_one"])


def test_explain_local_additivity_reconstructs_margin(fitted_bundle):
    """base_value + sum(contributions) must equal the model's raw margin output."""
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    results = explain_local(explainer, X, feature_names)
    margins = model.predict(X[feature_names], output_margin=True)

    for result, margin in zip(results, margins):
        assert result.predicted_margin == pytest.approx(float(margin), abs=1e-4)
        reconstructed = result.base_value + sum(result.feature_contributions.values())
        assert reconstructed == pytest.approx(result.predicted_margin, abs=1e-6)


def test_explain_local_feature_values_match_input(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    results = explain_local(explainer, X, feature_names)

    row0 = results[0]
    assert row0.feature_values["amount"] == pytest.approx(float(X.iloc[0]["amount"]))
    assert row0.feature_values["age"] == int(X.iloc[0]["age"])
    assert row0.feature_values["category"] == X.iloc[0]["category"]


def test_explain_local_ranked_features_sorted_by_abs_contribution(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    results = explain_local(explainer, X, feature_names)

    for result in results:
        abs_contribs = [abs(result.feature_contributions[name]) for name in result.ranked_features]
        assert abs_contribs == sorted(abs_contribs, reverse=True)


def test_explain_local_top_contributors_respects_n(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    results = explain_local(explainer, X, feature_names)
    top = results[0].top_contributors(n=2)
    assert len(top) == 2


def test_explain_local_to_dict_json_friendly_types(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    result = explain_local(explainer, X, feature_names)[0]
    d = result.to_dict()
    assert isinstance(d["feature_contributions"], dict)
    assert all(isinstance(v, float) for v in d["feature_contributions"].values())


def test_explain_local_never_includes_target_column(fused_bundle):
    model, fused, feature_names = fused_bundle
    explainer = create_tree_explainer(model)
    results = explain_local(explainer, fused, feature_names)
    for result in results:
        assert "FraudFound_P" not in result.feature_contributions
        assert "FraudFound_P" not in result.feature_values
        assert CLAIM_KEY_COL not in result.feature_contributions


# ---------------------------------------------------------------------------
# explain_one
# ---------------------------------------------------------------------------


def test_explain_one_single_row(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    result = explain_one(explainer, X.iloc[[0]], feature_names, claim_key="claim:0")
    assert isinstance(result, LocalExplanation)
    assert result.claim_key == "claim:0"


def test_explain_one_matches_explain_local(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    one = explain_one(explainer, X.iloc[[3]], feature_names)
    many = explain_local(explainer, X, feature_names)[3]
    assert one.feature_contributions == pytest.approx(many.feature_contributions)


def test_explain_one_multi_row_raises(fitted_bundle):
    model, X, feature_names = fitted_bundle
    explainer = create_tree_explainer(model)
    with pytest.raises(ExplainabilityError, match="single-row"):
        explain_one(explainer, X.iloc[0:2], feature_names)


# ---------------------------------------------------------------------------
# health-style and motor-style feature names both work identically
# ---------------------------------------------------------------------------


def test_consistent_result_structure_across_feature_sets():
    """Global/local result shape must not depend on which model's feature set is used."""
    rng = np.random.RandomState(1)
    motor_X = pd.DataFrame(
        {
            "PolicyAge": rng.randint(1, 20, size=30),
            "GarageID": pd.Categorical(rng.choice(["g1", "g2"], size=30)),
        }
    )
    health_X = pd.DataFrame(
        {
            "num_diagnosis_codes": rng.randint(1, 10, size=30),
            "chronic_condition_count": rng.randint(0, 5, size=30),
        }
    )
    motor_y = pd.Series((motor_X["PolicyAge"] > 10).astype(int))
    health_y = pd.Series((health_X["num_diagnosis_codes"] > 5).astype(int))

    motor_model = _fit_model(motor_X, motor_y)
    health_model = _fit_model(health_X, health_y)

    motor_explainer = create_tree_explainer(motor_model)
    health_explainer = create_tree_explainer(health_model)

    motor_global = explain_global(motor_explainer, motor_X, list(motor_X.columns))
    health_global = explain_global(health_explainer, health_X, list(health_X.columns))
    assert type(motor_global) is type(health_global) is GlobalExplanation

    motor_local = explain_local(motor_explainer, motor_X, list(motor_X.columns))
    health_local = explain_local(health_explainer, health_X, list(health_X.columns))
    assert type(motor_local[0]) is type(health_local[0]) is LocalExplanation
