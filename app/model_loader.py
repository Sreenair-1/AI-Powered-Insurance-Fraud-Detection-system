"""Persisted-model loading for the Streamlit app.

Wraps `src.models.persistence.load_health_model`/`load_motor_model` with an
app-facing error (`ModelNotAvailableError`) that tells the user exactly how
to produce an artifact (`scripts/train_models.py`), plus `st.cache_resource`
wrappers so a model is loaded from disk once per process, not once per
Streamlit rerun. The non-cached `load_*_model_or_raise` functions are the
testable core; the `get_*_model` wrappers are what `app/main.py` calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import streamlit as st

from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.models.persistence import ModelPersistenceError, load_health_model, load_motor_model

PathLike = Union[str, Path]

EVAL_METRICS_FILENAME = "eval_metrics.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.json"
ROC_CURVE_FILENAME = "roc_curve.json"
PR_CURVE_FILENAME = "pr_curve.json"
TEST_METRICS_FILENAME = "test_metrics.json"
THRESHOLD_SELECTION_FILENAME = "threshold_selection.json"


class ModelNotAvailableError(RuntimeError):
    """Raised when a persisted model artifact cannot be loaded for the app."""


def load_health_model_or_raise(directory: PathLike) -> HealthFraudModel:
    """Load a fitted `HealthFraudModel`, or raise an actionable `ModelNotAvailableError`."""
    try:
        return load_health_model(directory)
    except ModelPersistenceError as exc:
        raise ModelNotAvailableError(
            f"No usable health model artifact at '{directory}'. Train and save one first, e.g. "
            "`./.venv/Scripts/python.exe scripts/train_models.py --domain health` "
            "(see docs/app_ui.md)."
        ) from exc


def load_motor_model_or_raise(directory: PathLike) -> MotorFraudModel:
    """Load a fitted `MotorFraudModel`, or raise an actionable `ModelNotAvailableError`."""
    try:
        return load_motor_model(directory)
    except ModelPersistenceError as exc:
        raise ModelNotAvailableError(
            f"No usable motor model artifact at '{directory}'. Train and save one first, e.g. "
            "`./.venv/Scripts/python.exe scripts/train_models.py --domain motor` "
            "(see docs/app_ui.md)."
        ) from exc


def load_eval_metrics(directory: PathLike) -> Optional[dict]:
    """Load the `eval_metrics.json` sidecar `scripts/train_models.py` writes, if present.

    This is a held-out validation-split evaluation captured at training
    time (Accuracy/Precision/Recall/F1/ROC-AUC), not something recomputed
    live against whatever sample the app happens to have loaded. Returns
    `None` if the file is absent -- an older/hand-built artifact directory
    may not have one, and that is not an error.
    """
    path = Path(directory) / EVAL_METRICS_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_confusion_matrix(directory: PathLike) -> Optional[dict]:
    """Load the `confusion_matrix.json` sidecar from training, if present.

    Counts are from the held-out validation split at training time, not from
    the interactive sample currently loaded in the app. Returns `None` when
    the sidecar is absent.
    """
    path = Path(directory) / CONFUSION_MATRIX_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_test_metrics(directory: PathLike) -> Optional[dict]:
    """Load separately persisted final test results, when present."""
    path = Path(directory) / TEST_METRICS_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_threshold_selection(directory: PathLike) -> Optional[dict]:
    """Load validation threshold objective and constraint metadata."""
    path = Path(directory) / THRESHOLD_SELECTION_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_roc_curve(directory: PathLike) -> Optional[dict]:
    """Load the `roc_curve.json` sidecar or embedded ROC curve data, if present.

    Returns a dict with 'fpr', 'tpr', and optional 'thresholds' lists, or None.
    """
    path = Path(directory) / ROC_CURVE_FILENAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    metrics = load_eval_metrics(directory)
    if metrics and isinstance(metrics.get("roc_curve"), dict):
        return metrics["roc_curve"]
    return None


def load_pr_curve(directory: PathLike) -> Optional[dict]:
    """Load the `pr_curve.json` sidecar or embedded PR curve data, if present.

    Returns a dict with 'precision', 'recall', and optional 'thresholds' lists, or None.
    """
    path = Path(directory) / PR_CURVE_FILENAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    metrics = load_eval_metrics(directory)
    if metrics and isinstance(metrics.get("pr_curve"), dict):
        return metrics["pr_curve"]
    return None


@st.cache_resource(show_spinner="Loading health fraud model...")
def get_health_model(directory: str) -> HealthFraudModel:
    return load_health_model_or_raise(directory)


@st.cache_resource(show_spinner="Loading motor fraud model...")
def get_motor_model(directory: str) -> MotorFraudModel:
    return load_motor_model_or_raise(directory)
