"""Unit tests for app/model_loader.py.

Tiny in-memory/synthetic fixtures only (mirrors tests/test_model_persistence.py),
no dependency on data/raw/ or a real trained artifact under models/.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.model_loader import (
    EVAL_METRICS_FILENAME,
    ModelNotAvailableError,
    get_health_model,
    get_motor_model,
    load_eval_metrics,
    load_health_model_or_raise,
    load_motor_model_or_raise,
)
from src.features.structured import CLAIM_KEY_COL
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.models.persistence import save_health_model, save_motor_model


def _synthetic_health_features(n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            CLAIM_KEY_COL: [f"claim:C{i}" for i in range(n)],
            "claim_type": pd.Series(rng.choice(["inpatient", "outpatient"], size=n)).astype("category"),
            "num_diagnosis_codes": rng.integers(0, 5, size=n),
            "degree_centrality": rng.random(n),
        }
    )
    y = pd.Series((rng.random(n) < 0.15).astype(int), name="target")
    if y.sum() == 0:
        y.iloc[0] = 1
    return X, y


def _synthetic_motor_features(n: int = 40, seed: int = 0):
    rng = np.random.RandomState(seed)
    numeric = rng.normal(size=n)
    categories = pd.Categorical(rng.choice(["A", "B", "C"], size=n))
    y_values = (numeric > 1.2).astype(int)
    if y_values.sum() == 0:
        y_values[0] = 1
    X = pd.DataFrame(
        {CLAIM_KEY_COL: [f"claim:{i}" for i in range(n)], "numeric_feature": numeric, "cat_feature": categories}
    )
    y = pd.Series(y_values, name="target")
    return X, y


@pytest.fixture
def saved_health_model_dir(tmp_path):
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    return save_health_model(model, tmp_path / "health_model"), X


@pytest.fixture
def saved_motor_model_dir(tmp_path):
    X, y = _synthetic_motor_features()
    model = MotorFraudModel().fit(X, y)
    return save_motor_model(model, tmp_path / "motor_model"), X


# ---------------------------------------------------------------------------
# load_*_model_or_raise
# ---------------------------------------------------------------------------


def test_load_health_model_or_raise_returns_usable_model(saved_health_model_dir):
    directory, X = saved_health_model_dir
    model = load_health_model_or_raise(directory)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_load_motor_model_or_raise_returns_usable_model(saved_motor_model_dir):
    directory, X = saved_motor_model_dir
    model = load_motor_model_or_raise(directory)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)


def test_load_health_model_or_raise_missing_dir_raises_actionable_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="train_models.py"):
        load_health_model_or_raise(tmp_path / "does_not_exist")


def test_load_motor_model_or_raise_missing_dir_raises_actionable_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="train_models.py"):
        load_motor_model_or_raise(tmp_path / "does_not_exist")


def test_load_health_model_or_raise_rejects_motor_artifact(saved_motor_model_dir):
    directory, _ = saved_motor_model_dir
    with pytest.raises(ModelNotAvailableError):
        load_health_model_or_raise(directory)


# ---------------------------------------------------------------------------
# eval metrics sidecar
# ---------------------------------------------------------------------------


def test_load_eval_metrics_returns_none_when_absent(saved_health_model_dir):
    directory, _ = saved_health_model_dir
    assert load_eval_metrics(directory) is None


def test_load_eval_metrics_reads_written_json(saved_health_model_dir):
    directory, _ = saved_health_model_dir
    payload = {"accuracy": 0.9, "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.8}
    (directory / EVAL_METRICS_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    assert load_eval_metrics(directory) == payload


# ---------------------------------------------------------------------------
# cached wrappers
# ---------------------------------------------------------------------------


def test_get_health_model_caches_by_directory(saved_health_model_dir):
    directory, _ = saved_health_model_dir
    a = get_health_model(str(directory))
    b = get_health_model(str(directory))
    assert a is b


def test_get_motor_model_caches_by_directory(saved_motor_model_dir):
    directory, _ = saved_motor_model_dir
    a = get_motor_model(str(directory))
    b = get_motor_model(str(directory))
    assert a is b


def test_get_health_model_raises_for_missing_directory(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        get_health_model(str(tmp_path / "nope"))
