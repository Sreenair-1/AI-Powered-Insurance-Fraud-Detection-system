"""Shared classification evaluation metrics for fraud risk models.

Scores both `HealthFraudModel` (`src.models.health`) and `MotorFraudModel`
(`src.models.motor`) on held-out val/test splits with a consistent metric
set: Accuracy, Precision, Recall, F1, ROC-AUC.

These are model-behavior diagnostics, not a fraud determination — a low
precision or high false-positive rate describes the model's performance on
a split, not any individual claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Accuracy/Precision/Recall/F1/ROC-AUC for one evaluation run."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float
    n_samples: int
    n_positive: int

    def to_dict(self) -> dict[str, float]:
        """Flat dict representation, e.g. for logging or a Streamlit table."""
        return asdict(self)


class FraudRiskModel(Protocol):
    """Structural type both `HealthFraudModel` and `MotorFraudModel` satisfy."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def compute_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_proba: Union[pd.Series, np.ndarray],
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute Accuracy/Precision/Recall/F1/ROC-AUC from labels and predicted probabilities.

    The predicted label used for accuracy/precision/recall/F1 is derived
    here as `y_proba >= threshold`, rather than taken from a model's own
    `.predict()` — `HealthFraudModel.predict()` and
    `MotorFraudModel.predict()` have different signatures (only the latter
    accepts a `threshold`), so deriving labels from probabilities in one
    place keeps the same threshold applied consistently across both models
    and across every metric in a single call.

    Args:
        y_true: True binary (0/1) labels.
        y_proba: Predicted positive-class probabilities, same length/order
            as `y_true` (e.g. `model.predict_proba(X)`).
        threshold: Probability cutoff for the positive class, in `[0, 1]`.

    Raises:
        ValueError: if the inputs are empty, mismatched in length,
            `threshold` is outside `[0, 1]`, `y_true` is not binary (0/1),
            or `y_true` contains only one class (ROC-AUC and
            precision/recall are undefined in that case).
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true has no samples")
    if y_true_arr.shape[0] != y_proba_arr.shape[0]:
        raise ValueError(
            f"y_true has {y_true_arr.shape[0]} samples but y_proba has {y_proba_arr.shape[0]}"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    unique = set(np.unique(y_true_arr).tolist())
    if not unique <= {0, 1}:
        raise ValueError(f"y_true must be binary (0/1); got values {sorted(unique)}")
    if len(unique) < 2:
        raise ValueError(
            "y_true must contain both classes (0 and 1) to compute these metrics; "
            f"got only {sorted(unique)}"
        )

    y_pred = (y_proba_arr >= threshold).astype(int)

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true_arr, y_pred)),
        precision=float(precision_score(y_true_arr, y_pred, zero_division=0)),
        recall=float(recall_score(y_true_arr, y_pred, zero_division=0)),
        f1=float(f1_score(y_true_arr, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true_arr, y_proba_arr)),
        threshold=threshold,
        n_samples=int(y_true_arr.shape[0]),
        n_positive=int((y_true_arr == 1).sum()),
    )


def evaluate_model(
    model: FraudRiskModel,
    X: pd.DataFrame,
    y_true: Union[pd.Series, np.ndarray],
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Score a fitted `HealthFraudModel` or `MotorFraudModel` on `(X, y_true)`.

    Calls `model.predict_proba(X)` and scores the result with
    `compute_classification_metrics`. `X`/`y_true` should be a val or test
    split — never the data the model was fit on.
    """
    y_proba = model.predict_proba(X)
    return compute_classification_metrics(y_true, y_proba, threshold=threshold)
