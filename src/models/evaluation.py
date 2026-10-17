"""Binary fraud-risk evaluation and validation-only threshold selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float
    n_samples: int
    n_positive: int
    average_precision: float = 0.0
    specificity: float = 0.0
    balanced_accuracy: float = 0.0
    review_volume: int = 0
    review_rate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ConfusionMatrix:
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    threshold: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    objective: str
    constraints: dict[str, float | int]
    metrics: ClassificationMetrics

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "objective": self.objective,
                "constraints": self.constraints, "metrics": self.metrics.to_dict()}


class FraudRiskModel(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def _validate(y_true, y_proba, threshold=0.5):
    y = np.asarray(y_true)
    p = np.asarray(y_proba, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or not len(y):
        raise ValueError("y_true and y_proba must be non-empty one-dimensional arrays")
    if len(y) != len(p):
        raise ValueError(f"y_true has {len(y)} samples but y_proba has {len(p)}")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("y_proba must contain finite probabilities in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError("y_true must be binary (0/1)")
    if len(np.unique(y)) < 2:
        raise ValueError("y_true must contain both classes (0 and 1)")
    return y.astype(int), p


def compute_classification_metrics(y_true, y_proba, threshold=0.5) -> ClassificationMetrics:
    y, p = _validate(y_true, y_proba, threshold)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    volume = int(pred.sum())
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, p)), threshold=float(threshold),
        n_samples=len(y), n_positive=int(y.sum()),
        average_precision=float(average_precision_score(y, p)),
        specificity=float(tn / (tn + fp)) if tn + fp else 0.0,
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        review_volume=volume, review_rate=volume / len(y),
    )


def compute_confusion_matrix(y_true, y_proba, threshold=0.5) -> ConfusionMatrix:
    y, p = _validate(y_true, y_proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    return ConfusionMatrix(int(tn), int(fp), int(fn), int(tp), float(threshold))


def select_threshold(
    y_true, y_proba, objective: Literal["max_f1", "recall_at_precision", "precision_at_recall", "review_capacity"] = "max_f1",
    *, min_precision: float | None = None, min_recall: float | None = None,
    review_capacity: int | None = None,
) -> ThresholdSelection:
    """Select a probability threshold from validation predictions only.

    Ties are resolved toward the higher threshold (fewer reviews). For a
    capacity objective, all claims at the cutoff probability are included.
    """
    y, p = _validate(y_true, y_proba, 0.5)
    if objective not in {"max_f1", "recall_at_precision", "precision_at_recall", "review_capacity"}:
        raise ValueError(f"unsupported threshold objective: {objective}")
    if objective == "recall_at_precision" and min_precision is None:
        raise ValueError("min_precision is required for recall_at_precision")
    if objective == "precision_at_recall" and min_recall is None:
        raise ValueError("min_recall is required for precision_at_recall")
    if min_precision is not None and not 0 <= min_precision <= 1:
        raise ValueError("min_precision must be in [0, 1]")
    if min_recall is not None and not 0 <= min_recall <= 1:
        raise ValueError("min_recall must be in [0, 1]")
    if objective == "review_capacity" and (review_capacity is None or not 0 <= review_capacity <= len(y)):
        raise ValueError(f"review_capacity must be between 0 and {len(y)}")

    thresholds = np.unique(np.r_[p, 0.0, 1.0])
    candidates = [(float(t), compute_classification_metrics(y, p, float(t))) for t in thresholds]
    if objective == "max_f1":
        selected = max(candidates, key=lambda x: (x[1].f1, x[0]))
    elif objective == "recall_at_precision":
        feasible = [x for x in candidates if x[1].precision >= min_precision]
        if not feasible:
            raise ValueError(f"infeasible threshold constraint: precision >= {min_precision}")
        selected = max(feasible, key=lambda x: (x[1].recall, x[1].precision, x[0]))
    elif objective == "precision_at_recall":
        feasible = [x for x in candidates if x[1].recall >= min_recall]
        if not feasible:
            raise ValueError(f"infeasible threshold constraint: recall >= {min_recall}")
        selected = max(feasible, key=lambda x: (x[1].precision, x[1].recall, x[0]))
    else:
        feasible = [x for x in candidates if x[1].review_volume <= review_capacity]
        if not feasible:
            raise ValueError(
                f"infeasible threshold constraint: review_capacity={review_capacity}; "
                "tied probabilities prevent meeting the requested capacity"
            )
        selected = max(feasible, key=lambda x: (x[1].recall, x[1].precision, x[0]))
    constraints: dict[str, float | int] = {}
    if min_precision is not None: constraints["min_precision"] = min_precision
    if min_recall is not None: constraints["min_recall"] = min_recall
    if review_capacity is not None: constraints["review_capacity"] = review_capacity
    return ThresholdSelection(selected[0], objective, constraints, selected[1])


def curve_data(y_true, y_proba) -> tuple[dict, dict]:
    y, p = _validate(y_true, y_proba, 0.5)
    fpr, tpr, roc_thresholds = roc_curve(y, p)
    precision, recall, pr_thresholds = precision_recall_curve(y, p)
    return (
        {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": roc_thresholds.tolist()},
        {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": pr_thresholds.tolist()},
    )


def evaluate_model(model: FraudRiskModel, X: pd.DataFrame, y_true, threshold=0.5) -> ClassificationMetrics:
    return compute_classification_metrics(y_true, model.predict_proba(X), threshold)


def evaluate_confusion_matrix(model: FraudRiskModel, X: pd.DataFrame, y_true, threshold=0.5) -> ConfusionMatrix:
    return compute_confusion_matrix(y_true, model.predict_proba(X), threshold)
