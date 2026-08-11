"""Save/load APIs for fitted `HealthFraudModel`/`MotorFraudModel` instances.

The underlying `xgboost.XGBClassifier` is serialized with XGBoost's own
JSON format (`XGBClassifier.save_model`/`load_model`), never `pickle` —
loading an artifact never executes arbitrary code, unlike an unpickle of
untrusted data. A small sidecar `metadata.json`, written with the stdlib
`json` module, carries the wrapper-level state XGBoost's own format doesn't
know about (fitted feature order, and — for motor — the fit-time
categorical dtype set). Round-tripping a fitted model through save/load
does not refit or otherwise alter it: `predict_proba` output is bit-for-bit
identical before and after (verified in
`tests/test_model_persistence.py`).

Each artifact is a directory with a fixed two-file layout:
    <directory>/model.json     -- XGBClassifier.save_model() output
    <directory>/metadata.json  -- feature names / categorical dtypes / params
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import pandas as pd
from xgboost import XGBClassifier

from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel

ARTIFACT_VERSION = 1

MODEL_FILENAME = "model.json"
METADATA_FILENAME = "metadata.json"

_HEALTH_MODEL_TYPE = "health"
_MOTOR_MODEL_TYPE = "motor"

PathLike = Union[str, Path]


class ModelPersistenceError(ValueError):
    """Raised for an invalid save target, an unfit model, or a corrupt/incompatible artifact."""


def _ensure_json_safe(value: Any, *, context: str) -> Any:
    try:
        json.dumps(value)
    except TypeError as exc:
        raise ModelPersistenceError(f"{context} is not JSON-serializable: {exc}") from exc
    return value


def _write_metadata(directory: Path, metadata: dict[str, Any]) -> None:
    (directory / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _read_metadata(directory: Path, *, expected_model_type: str) -> dict[str, Any]:
    metadata_path = directory / METADATA_FILENAME
    model_path = directory / MODEL_FILENAME
    if not metadata_path.exists():
        raise ModelPersistenceError(f"missing metadata file: {metadata_path}")
    if not model_path.exists():
        raise ModelPersistenceError(f"missing model file: {model_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelPersistenceError(f"corrupt metadata file {metadata_path}: {exc}") from exc
    if metadata.get("model_type") != expected_model_type:
        raise ModelPersistenceError(
            f"artifact at {directory} has model_type={metadata.get('model_type')!r}, "
            f"expected {expected_model_type!r}"
        )
    if metadata.get("version") != ARTIFACT_VERSION:
        raise ModelPersistenceError(
            f"artifact at {directory} has version={metadata.get('version')!r}, "
            f"expected {ARTIFACT_VERSION!r} (incompatible persistence format)"
        )
    return metadata


def save_health_model(model: HealthFraudModel, directory: PathLike) -> Path:
    """Save a fitted `HealthFraudModel` to `directory` (created if absent).

    Raises:
        ModelPersistenceError: if `model` has not been fit yet.
    """
    if model.model is None or model.feature_names_ is None:
        raise ModelPersistenceError("cannot save an unfit HealthFraudModel; call fit() first")

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_model(str(out_dir / MODEL_FILENAME))

    metadata = {
        "version": ARTIFACT_VERSION,
        "model_type": _HEALTH_MODEL_TYPE,
        "feature_names": list(model.feature_names_),
        "params": _ensure_json_safe(model._params, context="HealthFraudModel params"),
        "explicit_scale_pos_weight": model._explicit_scale_pos_weight,
    }
    _write_metadata(out_dir, metadata)
    return out_dir


def load_health_model(directory: PathLike) -> HealthFraudModel:
    """Load a `HealthFraudModel` previously saved with `save_health_model`.

    The returned model is ready for `predict`/`predict_proba` immediately —
    it is not refit, and its predictions match the original fitted model
    exactly.
    """
    in_dir = Path(directory)
    metadata = _read_metadata(in_dir, expected_model_type=_HEALTH_MODEL_TYPE)

    booster = XGBClassifier()
    booster.load_model(str(in_dir / MODEL_FILENAME))

    model = HealthFraudModel()
    model.model = booster
    model.feature_names_ = list(metadata["feature_names"])
    model._params = dict(metadata["params"])
    model._explicit_scale_pos_weight = metadata.get("explicit_scale_pos_weight")
    return model


def save_motor_model(model: MotorFraudModel, directory: PathLike) -> Path:
    """Save a fitted `MotorFraudModel` to `directory` (created if absent).

    Raises:
        ModelPersistenceError: if `model` has not been fit yet.
    """
    if model._model is None or model._feature_columns is None:
        raise ModelPersistenceError("cannot save an unfit MotorFraudModel; call fit() first")

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    model._model.save_model(str(out_dir / MODEL_FILENAME))

    categorical_dtypes = {
        col: {
            "categories": _ensure_json_safe(
                dtype.categories.tolist(), context=f"categorical dtype for column {col!r}"
            ),
            "ordered": bool(dtype.ordered),
        }
        for col, dtype in model._categorical_dtypes.items()
    }
    metadata = {
        "version": ARTIFACT_VERSION,
        "model_type": _MOTOR_MODEL_TYPE,
        "feature_columns": list(model._feature_columns),
        "categorical_dtypes": categorical_dtypes,
        "params": _ensure_json_safe(model.params, context="MotorFraudModel params"),
        "scale_pos_weight": model._scale_pos_weight,
    }
    _write_metadata(out_dir, metadata)
    return out_dir


def load_motor_model(directory: PathLike) -> MotorFraudModel:
    """Load a `MotorFraudModel` previously saved with `save_motor_model`.

    The returned model is ready for `predict`/`predict_proba` immediately —
    it is not refit, and its predictions match the original fitted model
    exactly, including unseen-category-to-missing handling (the fit-time
    categorical dtype set is restored from the metadata sidecar).
    """
    in_dir = Path(directory)
    metadata = _read_metadata(in_dir, expected_model_type=_MOTOR_MODEL_TYPE)

    booster = XGBClassifier()
    booster.load_model(str(in_dir / MODEL_FILENAME))

    model = MotorFraudModel(params=dict(metadata["params"]))
    model._model = booster
    model._feature_columns = list(metadata["feature_columns"])
    model._categorical_dtypes = {
        col: pd.CategoricalDtype(categories=spec["categories"], ordered=spec["ordered"])
        for col, spec in metadata["categorical_dtypes"].items()
    }
    model._scale_pos_weight = metadata.get("scale_pos_weight")
    return model
