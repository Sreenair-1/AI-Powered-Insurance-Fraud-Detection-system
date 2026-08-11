"""Unit tests for src/models/persistence.py: save/load for HealthFraudModel
and MotorFraudModel.

Tiny in-memory/synthetic fixtures only, no data/raw/ dependency. Uses
pytest's tmp_path for all on-disk artifacts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.features.structured import CLAIM_KEY_COL
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.models.persistence import (
    ModelPersistenceError,
    load_health_model,
    load_motor_model,
    save_health_model,
    save_motor_model,
)


# ---------------------------------------------------------------------------
# Fixtures (mirrors tests/test_model_health.py / tests/test_model_motor.py)
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


def _tiny_motor_XY(n=40, seed=0):
    rng = np.random.RandomState(seed)
    numeric = rng.normal(size=n)
    categories = pd.Categorical(rng.choice(["A", "B", "C"], size=n))
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


# ---------------------------------------------------------------------------
# Health model save/load
# ---------------------------------------------------------------------------


def test_health_save_creates_expected_files(tmp_path):
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    out_dir = save_health_model(model, tmp_path / "health_model")
    assert (out_dir / "model.json").exists()
    assert (out_dir / "metadata.json").exists()


def test_health_save_rejects_unfit_model(tmp_path):
    with pytest.raises(ModelPersistenceError):
        save_health_model(HealthFraudModel(), tmp_path / "health_model")


def test_health_load_predictions_match_original(tmp_path):
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    save_health_model(model, tmp_path / "health_model")

    loaded = load_health_model(tmp_path / "health_model")
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_health_load_preserves_feature_names(tmp_path):
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    save_health_model(model, tmp_path / "health_model")

    loaded = load_health_model(tmp_path / "health_model")
    assert loaded.feature_names_ == model.feature_names_


def test_health_load_on_holdout_rows(tmp_path):
    X, y = _synthetic_health_features(n=40, seed=1)
    X_train, y_train = X.iloc[:28].reset_index(drop=True), y.iloc[:28].reset_index(drop=True)
    X_val = X.iloc[28:].reset_index(drop=True)
    model = HealthFraudModel().fit(X_train, y_train)
    save_health_model(model, tmp_path / "health_model")

    loaded = load_health_model(tmp_path / "health_model")
    np.testing.assert_allclose(loaded.predict_proba(X_val), model.predict_proba(X_val))


def test_health_load_missing_directory_raises(tmp_path):
    with pytest.raises(ModelPersistenceError):
        load_health_model(tmp_path / "does_not_exist")


def test_health_load_rejects_motor_artifact(tmp_path):
    X, y = _tiny_motor_XY()
    motor_model = MotorFraudModel().fit(X, y)
    save_motor_model(motor_model, tmp_path / "motor_model")

    with pytest.raises(ModelPersistenceError):
        load_health_model(tmp_path / "motor_model")


def test_health_load_rejects_corrupt_metadata(tmp_path):
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    out_dir = save_health_model(model, tmp_path / "health_model")
    (out_dir / "metadata.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ModelPersistenceError):
        load_health_model(out_dir)


def test_health_load_rejects_incompatible_version(tmp_path):
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    out_dir = save_health_model(model, tmp_path / "health_model")
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["version"] = 999
    (out_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModelPersistenceError):
        load_health_model(out_dir)


def test_health_save_creates_parent_directories(tmp_path):
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    nested = tmp_path / "a" / "b" / "c" / "health_model"
    save_health_model(model, nested)
    assert nested.exists()


# ---------------------------------------------------------------------------
# Motor model save/load
# ---------------------------------------------------------------------------


def test_motor_save_creates_expected_files(tmp_path):
    X, y = _tiny_motor_XY()
    model = MotorFraudModel().fit(X, y)
    out_dir = save_motor_model(model, tmp_path / "motor_model")
    assert (out_dir / "model.json").exists()
    assert (out_dir / "metadata.json").exists()


def test_motor_save_rejects_unfit_model(tmp_path):
    with pytest.raises(ModelPersistenceError):
        save_motor_model(MotorFraudModel(), tmp_path / "motor_model")


def test_motor_load_predictions_match_original(tmp_path):
    X, y = _tiny_motor_XY()
    model = MotorFraudModel().fit(X, y)
    save_motor_model(model, tmp_path / "motor_model")

    loaded = load_motor_model(tmp_path / "motor_model")
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_motor_load_preserves_feature_columns_and_scale_pos_weight(tmp_path):
    X, y = _tiny_motor_XY()
    model = MotorFraudModel().fit(X, y)
    save_motor_model(model, tmp_path / "motor_model")

    loaded = load_motor_model(tmp_path / "motor_model")
    assert loaded.feature_columns == model.feature_columns
    assert loaded.scale_pos_weight == pytest.approx(model.scale_pos_weight)


def test_motor_load_preserves_categorical_dtype_and_unseen_category_handling(tmp_path):
    X, y = _tiny_motor_XY(n=60, seed=3)
    model = MotorFraudModel().fit(X, y)
    save_motor_model(model, tmp_path / "motor_model")
    loaded = load_motor_model(tmp_path / "motor_model")

    X_new = X.copy()
    X_new["cat_feature"] = X_new["cat_feature"].cat.add_categories(["Z"])
    X_new.loc[X_new.index[0], "cat_feature"] = "Z"

    original_proba = model.predict_proba(X_new)
    loaded_proba = loaded.predict_proba(X_new)
    np.testing.assert_allclose(loaded_proba, original_proba)
    assert np.all(np.isfinite(loaded_proba))


def test_motor_load_missing_directory_raises(tmp_path):
    with pytest.raises(ModelPersistenceError):
        load_motor_model(tmp_path / "does_not_exist")


def test_motor_load_rejects_health_artifact(tmp_path):
    X, y = _synthetic_health_features()
    health_model = HealthFraudModel().fit(X, y)
    save_health_model(health_model, tmp_path / "health_model")

    with pytest.raises(ModelPersistenceError):
        load_motor_model(tmp_path / "health_model")


def test_motor_load_missing_model_file_raises(tmp_path):
    X, y = _tiny_motor_XY()
    model = MotorFraudModel().fit(X, y)
    out_dir = save_motor_model(model, tmp_path / "motor_model")
    (out_dir / "model.json").unlink()

    with pytest.raises(ModelPersistenceError):
        load_motor_model(out_dir)


def test_motor_save_creates_parent_directories(tmp_path):
    X, y = _tiny_motor_XY()
    model = MotorFraudModel().fit(X, y)
    nested = tmp_path / "x" / "y" / "motor_model"
    save_motor_model(model, nested)
    assert nested.exists()


# ---------------------------------------------------------------------------
# End-to-end: fuse_motor_claims() -> fit -> save -> load -> predict
# ---------------------------------------------------------------------------


def test_motor_end_to_end_save_load_via_fuse_motor_claims(tmp_path):
    from src.data.schemas import MOTOR_ALL_COLUMNS, MOTOR_TARGET
    from src.data.synthetic import FraudRingConfig, assign_synthetic_entities, inject_fraud_rings
    from src.features import fusion
    from src.graph import construction

    claims = pd.DataFrame({c: 0 for c in MOTOR_ALL_COLUMNS}, index=range(30))
    claims["Make"] = "Honda"
    claims[MOTOR_TARGET] = 0
    with_entities = assign_synthetic_entities(claims, seed=42, garage_pool_ratio=0.1, bank_pool_ratio=0.1)
    augmented, _ = inject_fraud_rings(
        with_entities, config=FraudRingConfig(num_rings=1, ring_size=3, seed=42)
    )
    augmented.loc[augmented.index[:2], MOTOR_TARGET] = 1

    graph = construction.build_motor_graph(augmented)
    X, y = fusion.fuse_motor_claims(augmented, graph)

    model = MotorFraudModel().fit(X, y)
    save_motor_model(model, tmp_path / "motor_model")
    loaded = load_motor_model(tmp_path / "motor_model")

    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
