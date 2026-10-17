import numpy as np
import pytest

from src.models.evaluation import compute_classification_metrics, select_threshold


def test_metrics_include_ap_specificity_balanced_accuracy_and_review_volume():
    metrics = compute_classification_metrics([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9], 0.5)
    assert metrics.average_precision == pytest.approx(5 / 6)
    assert metrics.specificity == pytest.approx(0.5)
    assert metrics.balanced_accuracy == pytest.approx(0.75)
    assert metrics.review_volume == 3
    assert metrics.review_rate == pytest.approx(0.75)


def test_max_f1_selects_threshold_from_validation_scores():
    result = select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], "max_f1")
    assert result.threshold == pytest.approx(0.6)
    assert result.metrics.f1 == pytest.approx(1.0)


def test_recall_at_minimum_precision_and_precision_at_minimum_recall():
    y = [0, 0, 1, 1]
    p = [0.1, 0.4, 0.6, 0.9]
    recall = select_threshold(y, p, "recall_at_precision", min_precision=1.0)
    precision = select_threshold(y, p, "precision_at_recall", min_recall=0.5)
    assert recall.metrics.precision >= 1.0
    assert recall.metrics.recall == pytest.approx(1.0)
    assert precision.metrics.recall >= 0.5
    assert precision.metrics.precision == pytest.approx(1.0)


def test_review_capacity_and_infeasible_constraint_are_explicit():
    selected = select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], "review_capacity", review_capacity=1)
    assert selected.metrics.review_volume <= 1
    with pytest.raises(ValueError, match="infeasible"):
        select_threshold([1, 0], [0.1, 0.9], "recall_at_precision", min_precision=1.0)


def test_invalid_threshold_objective_parameters_raise():
    with pytest.raises(ValueError, match="required"):
        select_threshold([0, 1], [0.1, 0.9], "recall_at_precision")
