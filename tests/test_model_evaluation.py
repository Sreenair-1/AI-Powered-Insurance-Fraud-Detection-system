"""Unit tests for src/models/evaluation.py: shared classification metrics.

Tiny synthetic/in-memory fixtures only, no data/raw/ dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.structured import CLAIM_KEY_COL
from src.models.evaluation import (
    ClassificationMetrics,
    ConfusionMatrix,
    compute_classification_metrics,
    compute_confusion_matrix,
    evaluate_confusion_matrix,
    evaluate_model,
)
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel


# ---------------------------------------------------------------------------
# compute_classification_metrics — correctness on known inputs
# ---------------------------------------------------------------------------


def test_perfect_predictions_score_1_0_on_every_metric():
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    metrics = compute_classification_metrics(y_true, y_proba)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)


def test_worst_case_predictions_score_0_on_precision_recall_f1_and_auc():
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.95, 0.9, 0.8, 0.2, 0.1, 0.05])
    metrics = compute_classification_metrics(y_true, y_proba)
    assert metrics.accuracy == pytest.approx(0.0)
    assert metrics.precision == pytest.approx(0.0)
    assert metrics.recall == pytest.approx(0.0)
    assert metrics.f1 == pytest.approx(0.0)
    assert metrics.roc_auc == pytest.approx(0.0)


def test_known_confusion_matrix_matches_hand_computed_metrics():
    # y_true: 5 negatives, 3 positives. At threshold 0.5:
    # TP=2, FN=1, FP=1, TN=4 -> accuracy=6/8, precision=2/3, recall=2/3, f1=2/3
    y_true = pd.Series([0, 0, 0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.3])
    metrics = compute_classification_metrics(y_true, y_proba)
    assert metrics.accuracy == pytest.approx(6 / 8)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)


def test_threshold_changes_predicted_labels():
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.3, 0.4, 0.6, 0.45])
    low = compute_classification_metrics(y_true, y_proba, threshold=0.2)
    high = compute_classification_metrics(y_true, y_proba, threshold=0.9)
    assert low.recall == pytest.approx(1.0)
    assert high.recall == pytest.approx(0.0)


def test_n_samples_and_n_positive_recorded():
    y_true = pd.Series([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
    metrics = compute_classification_metrics(y_true, y_proba)
    assert metrics.n_samples == 5
    assert metrics.n_positive == 2
    assert metrics.threshold == pytest.approx(0.5)


def test_returns_classification_metrics_instance_and_to_dict():
    y_true = pd.Series([0, 1])
    y_proba = np.array([0.1, 0.9])
    metrics = compute_classification_metrics(y_true, y_proba)
    assert isinstance(metrics, ClassificationMetrics)
    d = metrics.to_dict()
    assert set(d.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "threshold",
        "n_samples",
        "n_positive",
    }


# ---------------------------------------------------------------------------
# compute_classification_metrics — validation errors
# ---------------------------------------------------------------------------


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        compute_classification_metrics(pd.Series([], dtype=int), np.array([]))


def test_mismatched_length_raises():
    with pytest.raises(ValueError):
        compute_classification_metrics(pd.Series([0, 1, 1]), np.array([0.1, 0.9]))


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_out_of_range_threshold_raises(threshold):
    with pytest.raises(ValueError):
        compute_classification_metrics(pd.Series([0, 1]), np.array([0.1, 0.9]), threshold=threshold)


def test_non_binary_y_true_raises():
    with pytest.raises(ValueError):
        compute_classification_metrics(pd.Series([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


def test_single_class_y_true_raises():
    with pytest.raises(ValueError):
        compute_classification_metrics(pd.Series([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# evaluate_model — generic wrapper over any predict_proba(X) model
# ---------------------------------------------------------------------------


class _StubModel:
    """Minimal FraudRiskModel double: fixed predict_proba output."""

    def __init__(self, proba: np.ndarray) -> None:
        self._proba = proba

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._proba


def test_evaluate_model_uses_predict_proba():
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    y_true = pd.Series([0, 0, 1, 1])
    model = _StubModel(np.array([0.1, 0.2, 0.8, 0.9]))
    metrics = evaluate_model(model, X, y_true)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(1.0)


def test_evaluate_model_passes_through_threshold():
    X = pd.DataFrame({"a": [1, 2]})
    y_true = pd.Series([0, 1])
    model = _StubModel(np.array([0.4, 0.4]))
    metrics = evaluate_model(model, X, y_true, threshold=0.5)
    assert metrics.threshold == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# evaluate_model — integration with the real fitted models
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
    y = pd.Series((rng.random(n) < 0.3).astype(int), name="target")
    if y.sum() == 0:
        y.iloc[0] = 1
    if y.sum() == n:
        y.iloc[0] = 0
    return X, y


def _synthetic_motor_features(n: int = 40, seed: int = 0):
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
    y = pd.Series((rng.random(n) < 0.3).astype(int), name="target")
    if y.sum() == 0:
        y.iloc[0] = 1
    if y.sum() == n:
        y.iloc[0] = 0
    return X, y


def test_evaluate_model_with_fitted_health_model():
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.n_samples == len(X)
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_evaluate_model_with_fitted_motor_model():
    X, y = _synthetic_motor_features()
    model = MotorFraudModel().fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.n_samples == len(X)
    assert 0.0 <= metrics.roc_auc <= 1.0


# ---------------------------------------------------------------------------
# compute_confusion_matrix — correctness on known inputs
# ---------------------------------------------------------------------------


def test_confusion_matrix_matches_hand_computed_counts():
    # Same fixture as test_known_confusion_matrix_matches_hand_computed_metrics:
    # TP=2, FN=1, FP=1, TN=4.
    y_true = pd.Series([0, 0, 0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.3])
    cm = compute_confusion_matrix(y_true, y_proba)
    assert cm.true_negative == 4
    assert cm.false_positive == 1
    assert cm.false_negative == 1
    assert cm.true_positive == 2
    assert cm.threshold == pytest.approx(0.5)


def test_confusion_matrix_perfect_predictions():
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.2, 0.8, 0.9, 0.95])
    cm = compute_confusion_matrix(y_true, y_proba)
    assert (cm.true_negative, cm.false_positive, cm.false_negative, cm.true_positive) == (3, 0, 0, 3)


def test_confusion_matrix_counts_sum_to_n_samples():
    y_true = pd.Series([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
    cm = compute_confusion_matrix(y_true, y_proba)
    total = cm.true_negative + cm.false_positive + cm.false_negative + cm.true_positive
    assert total == len(y_true)


def test_confusion_matrix_threshold_changes_counts():
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.3, 0.4, 0.6, 0.45])
    low = compute_confusion_matrix(y_true, y_proba, threshold=0.2)
    high = compute_confusion_matrix(y_true, y_proba, threshold=0.9)
    assert low.false_negative == 0
    assert high.true_positive == 0


def test_confusion_matrix_returns_confusion_matrix_instance_and_to_dict():
    y_true = pd.Series([0, 1])
    y_proba = np.array([0.1, 0.9])
    cm = compute_confusion_matrix(y_true, y_proba)
    assert isinstance(cm, ConfusionMatrix)
    d = cm.to_dict()
    assert set(d.keys()) == {
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "threshold",
    }


# ---------------------------------------------------------------------------
# compute_confusion_matrix — validation errors (shared with compute_classification_metrics)
# ---------------------------------------------------------------------------


def test_confusion_matrix_empty_inputs_raise():
    with pytest.raises(ValueError):
        compute_confusion_matrix(pd.Series([], dtype=int), np.array([]))


def test_confusion_matrix_mismatched_length_raises():
    with pytest.raises(ValueError):
        compute_confusion_matrix(pd.Series([0, 1, 1]), np.array([0.1, 0.9]))


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_confusion_matrix_out_of_range_threshold_raises(threshold):
    with pytest.raises(ValueError):
        compute_confusion_matrix(pd.Series([0, 1]), np.array([0.1, 0.9]), threshold=threshold)


def test_confusion_matrix_non_binary_y_true_raises():
    with pytest.raises(ValueError):
        compute_confusion_matrix(pd.Series([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


def test_confusion_matrix_single_class_y_true_raises():
    with pytest.raises(ValueError):
        compute_confusion_matrix(pd.Series([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# evaluate_confusion_matrix — generic wrapper over any predict_proba(X) model
# ---------------------------------------------------------------------------


def test_evaluate_confusion_matrix_uses_predict_proba():
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    y_true = pd.Series([0, 0, 1, 1])
    model = _StubModel(np.array([0.1, 0.2, 0.8, 0.9]))
    cm = evaluate_confusion_matrix(model, X, y_true)
    assert (cm.true_negative, cm.false_positive, cm.false_negative, cm.true_positive) == (2, 0, 0, 2)


def test_evaluate_confusion_matrix_with_fitted_health_model():
    X, y = _synthetic_health_features()
    model = HealthFraudModel().fit(X, y)
    cm = evaluate_confusion_matrix(model, X, y)
    assert isinstance(cm, ConfusionMatrix)
    total = cm.true_negative + cm.false_positive + cm.false_negative + cm.true_positive
    assert total == len(X)


def test_evaluate_confusion_matrix_with_fitted_motor_model():
    X, y = _synthetic_motor_features()
    model = MotorFraudModel().fit(X, y)
    cm = evaluate_confusion_matrix(model, X, y)
    assert isinstance(cm, ConfusionMatrix)
    total = cm.true_negative + cm.false_positive + cm.false_negative + cm.true_positive
    assert total == len(X)


def test_confusion_matrix_and_classification_metrics_are_consistent():
    y_true = pd.Series([0, 0, 0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.3])
    cm = compute_confusion_matrix(y_true, y_proba)
    metrics = compute_classification_metrics(y_true, y_proba)
    n = cm.true_negative + cm.false_positive + cm.false_negative + cm.true_positive
    accuracy = (cm.true_negative + cm.true_positive) / n
    assert accuracy == pytest.approx(metrics.accuracy)
    precision = cm.true_positive / (cm.true_positive + cm.false_positive)
    assert precision == pytest.approx(metrics.precision)
    recall = cm.true_positive / (cm.true_positive + cm.false_negative)
    assert recall == pytest.approx(metrics.recall)
