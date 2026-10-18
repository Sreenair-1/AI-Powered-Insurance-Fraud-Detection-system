import numpy as np
import pytest

from src.models.evaluation import compute_classification_metrics, probability_distribution, select_threshold


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


def test_max_f1_does_not_choose_trivial_all_positive_threshold():
    result = select_threshold([0, 0, 1, 1], [0.2, 0.3, 0.31, 0.32], "max_f1")
    assert result.metrics.review_volume < 4
    assert result.metrics.review_volume > 0


def test_max_f1_rejects_near_all_positive_optimum_and_fixed_baseline_is_safe():
    y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    p = [.5] * len(y)
    with pytest.raises(ValueError, match="more than 90%"):
        select_threshold(y, p, "max_f1")
    fixed = select_threshold(y, p, "fixed_0_5")
    assert fixed.threshold == .5
    assert fixed.metrics.review_volume == 10


def test_constraints_compose_and_infeasibility_is_not_relaxed():
    selected = select_threshold([0, 0, 1, 1], [.1, .4, .6, .9], "recall_at_precision",
                                min_precision=.8, review_capacity=2)
    assert selected.metrics.precision >= .8
    assert selected.metrics.review_volume <= 2
    with pytest.raises(ValueError, match="infeasible threshold constraints"):
        select_threshold([0, 0, 1, 1], [.1, .4, .6, .9], "recall_at_precision",
                         min_precision=.8, min_recall=1, review_capacity=1)


def test_fixed_reference_threshold_does_not_silently_move_to_satisfy_constraints():
    with pytest.raises(ValueError, match="infeasible threshold constraints at fixed"):
        select_threshold([0, 0, 1, 1], [.1, .4, .6, .9], "fixed_0_5", review_capacity=1)


def test_probability_distribution_reports_quantiles_and_histogram():
    result = probability_distribution([.05, .25, .75, .95])
    assert result["count"] == 4
    assert result["quantiles"]["0.5"] == pytest.approx(.5)
    assert sum(result["histogram"]["counts"]) == 4
