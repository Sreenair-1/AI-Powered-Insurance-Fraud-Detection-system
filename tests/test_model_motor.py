"""Unit tests for src/models/motor.py: motor fraud XGBoost model.
Tiny in-memory/synthetic fixtures only, no data/raw/ dependency."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schemas import MOTOR_ALL_COLUMNS, MOTOR_TARGET
from src.data.synthetic import FraudRingConfig, assign_synthetic_entities, inject_fraud_rings
from src.features import fusion
from src.features.structured import CLAIM_KEY_COL, build_motor_structured_features
from src.graph import construction
from src.models.motor import MotorFraudModel, MotorModelError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_XY(n=40, seed=0):
    """A small, deterministic, imbalanced synthetic (X, y) pair with a
    claim_key column, one numeric feature, and one categorical feature —
    mirrors what build_motor_structured_features() produces."""
    rng = np.random.RandomState(seed)
    numeric = rng.normal(size=n)
    categories = pd.Categorical(rng.choice(["A", "B", "C"], size=n))
    # Make the label weakly dependent on the numeric feature so the model has
    # something learnable, and keep it imbalanced (few positives).
    y_values = (numeric > 1.2).astype(int)
    if y_values.sum() == 0:
        y_values[0] = 1
    X = pd.DataFrame(
        {
            CLAIM_KEY_COL: [f"claim:{i}" for i in range(n)],
            "numeric_feature": numeric,
            "cat_feature": categories,
        }
    )
    y = pd.Series(y_values, name="target")
    return X, y


def _augmented_motor_claims_df(n=30):
    claims = pd.DataFrame({c: 0 for c in MOTOR_ALL_COLUMNS}, index=range(n))
    claims["Make"] = "Honda"
    claims[MOTOR_TARGET] = 0
    with_entities = assign_synthetic_entities(claims, seed=42, garage_pool_ratio=0.1, bank_pool_ratio=0.1)
    augmented, _ = inject_fraud_rings(
        with_entities, config=FraudRingConfig(num_rings=1, ring_size=3, seed=42)
    )
    # Give a couple of extra rows the real FraudFound_P=1 label so y has both
    # classes (inject_fraud_rings never touches FraudFound_P).
    augmented.loc[augmented.index[:2], MOTOR_TARGET] = 1
    return augmented


# ---------------------------------------------------------------------------
# Basic fit/predict/predict_proba behavior
# ---------------------------------------------------------------------------


def test_fit_returns_self():
    X, y = _tiny_XY()
    model = MotorFraudModel()
    assert model.fit(X, y) is model


def test_predict_proba_shape_and_range():
    X, y = _tiny_XY()
    model = MotorFraudModel().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_predict_returns_binary_flags():
    X, y = _tiny_XY()
    model = MotorFraudModel().fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)) <= {0, 1}
    assert preds.shape == (len(X),)


def test_predict_threshold_zero_flags_everything():
    X, y = _tiny_XY()
    model = MotorFraudModel().fit(X, y)
    preds = model.predict(X, threshold=0.0)
    assert np.all(preds == 1)


def test_predict_threshold_one_is_valid_boundary():
    X, y = _tiny_XY()
    model = MotorFraudModel().fit(X, y)
    preds = model.predict(X, threshold=1.0)
    assert set(np.unique(preds)) <= {0, 1}


def test_predict_invalid_threshold_raises():
    X, y = _tiny_XY()
    model = MotorFraudModel().fit(X, y)
    with pytest.raises(MotorModelError):
        model.predict(X, threshold=1.5)
    with pytest.raises(MotorModelError):
        model.predict(X, threshold=-0.1)


# ---------------------------------------------------------------------------
# claim_key handling / target separation
# ---------------------------------------------------------------------------


def test_claim_key_dropped_from_model_features():
    X, y = _tiny_XY()
    model = MotorFraudModel().fit(X, y)
    assert CLAIM_KEY_COL not in model.feature_columns


def test_feature_columns_exclude_claim_key_and_match_input():
    X, y = _tiny_XY()
    model = MotorFraudModel().fit(X, y)
    assert model.feature_columns == ["numeric_feature", "cat_feature"]


# ---------------------------------------------------------------------------
# Class imbalance handling
# ---------------------------------------------------------------------------


def test_scale_pos_weight_computed_from_training_target():
    X, y = _tiny_XY(n=100, seed=1)
    model = MotorFraudModel().fit(X, y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    assert model.scale_pos_weight == pytest.approx(neg / pos)


def test_scale_pos_weight_none_before_fit():
    model = MotorFraudModel()
    assert model.scale_pos_weight is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_predictions_across_fits():
    X, y = _tiny_XY(n=60, seed=2)
    model_a = MotorFraudModel().fit(X, y)
    model_b = MotorFraudModel().fit(X, y)
    np.testing.assert_array_equal(model_a.predict_proba(X), model_b.predict_proba(X))


# ---------------------------------------------------------------------------
# Categorical dtype handling
# ---------------------------------------------------------------------------


def test_categorical_dtype_preserved_and_unseen_category_becomes_missing():
    X, y = _tiny_XY(n=60, seed=3)
    model = MotorFraudModel().fit(X, y)

    X_new = X.copy()
    # Introduce a category value never seen at fit time.
    X_new["cat_feature"] = X_new["cat_feature"].cat.add_categories(["Z"])
    X_new.loc[X_new.index[0], "cat_feature"] = "Z"

    proba = model.predict_proba(X_new)
    assert proba.shape == (len(X_new),)
    assert np.all(np.isfinite(proba))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_fit_rejects_non_dataframe_X():
    _, y = _tiny_XY()
    with pytest.raises(MotorModelError):
        MotorFraudModel().fit([[1, 2], [3, 4]], y)


def test_fit_rejects_empty_X():
    X, y = _tiny_XY()
    with pytest.raises(MotorModelError):
        MotorFraudModel().fit(X.iloc[0:0], y.iloc[0:0])


def test_fit_rejects_missing_claim_key_column():
    X, y = _tiny_XY()
    X_no_key = X.drop(columns=[CLAIM_KEY_COL])
    with pytest.raises(MotorModelError):
        MotorFraudModel().fit(X_no_key, y)


def test_fit_rejects_non_series_y():
    X, y = _tiny_XY()
    with pytest.raises(MotorModelError):
        MotorFraudModel().fit(X, y.to_numpy())


def test_fit_rejects_mismatched_length_y():
    X, y = _tiny_XY()
    with pytest.raises(MotorModelError):
        MotorFraudModel().fit(X, y.iloc[:-1])


def test_fit_rejects_y_with_missing_values():
    X, y = _tiny_XY()
    y_bad = y.astype(float)
    y_bad.iloc[0] = np.nan
    with pytest.raises(MotorModelError):
        MotorFraudModel().fit(X, y_bad)


def test_fit_rejects_non_binary_y():
    X, y = _tiny_XY()
    y_bad = y.copy()
    y_bad.iloc[0] = 2
    with pytest.raises(MotorModelError):
        MotorFraudModel().fit(X, y_bad)


def test_fit_rejects_single_class_y():
    X, y = _tiny_XY()
    y_single = pd.Series(np.zeros(len(y), dtype=int))
    with pytest.raises(MotorModelError):
        MotorFraudModel().fit(X, y_single)


def test_predict_before_fit_raises():
    X, _ = _tiny_XY()
    with pytest.raises(MotorModelError):
        MotorFraudModel().predict(X)


def test_predict_proba_before_fit_raises():
    X, _ = _tiny_XY()
    with pytest.raises(MotorModelError):
        MotorFraudModel().predict_proba(X)


def test_predict_missing_feature_column_raises():
    X, y = _tiny_XY()
    model = MotorFraudModel().fit(X, y)
    X_missing = X.drop(columns=["numeric_feature"])
    with pytest.raises(MotorModelError):
        model.predict(X_missing)


def test_predict_extra_feature_column_raises():
    X, y = _tiny_XY()
    model = MotorFraudModel().fit(X, y)
    X_extra = X.copy()
    X_extra["unexpected_feature"] = 0
    with pytest.raises(MotorModelError):
        model.predict(X_extra)


# ---------------------------------------------------------------------------
# Train/validation/test usage without fitting on val/test
# ---------------------------------------------------------------------------


def test_fit_only_on_train_then_predict_on_disjoint_val_and_test():
    X, y = _tiny_XY(n=90, seed=4)
    X_train, y_train = X.iloc[:60], y.iloc[:60]
    X_val = X.iloc[60:75]
    X_test = X.iloc[75:]

    model = MotorFraudModel().fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)
    test_proba = model.predict_proba(X_test)

    assert val_proba.shape == (len(X_val),)
    assert test_proba.shape == (len(X_test),)


# ---------------------------------------------------------------------------
# End-to-end integration with fuse_motor_claims()
# ---------------------------------------------------------------------------


def test_end_to_end_fuse_motor_claims_then_fit_predict():
    claims = _augmented_motor_claims_df()
    graph = construction.build_motor_graph(claims)
    X, y = fusion.fuse_motor_claims(claims, graph)

    model = MotorFraudModel().fit(X, y)
    proba = model.predict_proba(X)
    preds = model.predict(X)

    assert proba.shape == (len(X),)
    assert preds.shape == (len(X),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert model.scale_pos_weight == pytest.approx(int((y == 0).sum()) / int((y == 1).sum()))


def test_end_to_end_target_never_used_as_feature():
    claims = _augmented_motor_claims_df()
    graph = construction.build_motor_graph(claims)
    X, y = fusion.fuse_motor_claims(claims, graph)

    model = MotorFraudModel().fit(X, y)
    assert MOTOR_TARGET not in model.feature_columns
    assert "target" not in model.feature_columns
