"""Process-wide state: trained models (loaded once) and scored sample datasets.

Models are only ever *loaded* from `models/<domain>/` (or the
`FRAUD_APP_*_MODEL_DIR` overrides in `app.config`) -- never fitted here.
Sampled datasets are built with the same `app.data_loader` builders and
seeds as the Streamlit app and cached per sample-size combination, the API
equivalent of Streamlit's `st.cache_data`.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from api.reviews import ReviewStore
from app import config, data_loader
from app.claim_view import build_investigation_table, score_loaded_sample
from app.model_loader import (
    ModelNotAvailableError,
    load_health_model_or_raise,
    load_motor_model_or_raise,
)
from src.graph import BANK_ACCOUNT_NODE_TYPE, BENEFICIARY_NODE_TYPE, GARAGE_NODE_TYPE, PHYSICIAN_NODE_TYPE
from src.graph.rings import (
    RingDetectionResult,
    RingDetectorConfig,
    claim_entities_from_graph,
    detect_candidate_rings,
)
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.pipeline.inference import DEFAULT_RISK_THRESHOLDS

HEALTH_DOMAIN = "health"
MOTOR_DOMAIN = "motor"
DOMAINS = (HEALTH_DOMAIN, MOTOR_DOMAIN)

# Entity types the ring detector looks for unexpected co-sharing between.
# Health providers are left out: every physician belongs to a provider, so
# that pair co-occurs by structure, not by chance.
NETWORK_ENTITY_TYPES = {
    MOTOR_DOMAIN: (GARAGE_NODE_TYPE, BANK_ACCOUNT_NODE_TYPE),
    HEALTH_DOMAIN: (PHYSICIAN_NODE_TYPE, BENEFICIARY_NODE_TYPE),
}
# Graph node ids are exact system identifiers, so no typo tolerance is needed.
NETWORK_DETECTOR = RingDetectorConfig(matching="raw")

Model = Union[HealthFraudModel, MotorFraudModel]
Dataset = Union[data_loader.HealthDataset, data_loader.MotorDataset]

# Sampled datasets kept in memory at once (each is a few MB at default sizes).
MAX_CACHED_SAMPLES = 6


class ModelUnavailable(RuntimeError):
    """The domain's trained model could not be loaded at startup."""


class DataUnavailable(RuntimeError):
    """The raw claims data for a domain is missing on disk."""


@dataclass(frozen=True)
class SampleParams:
    """Sample sizes for one domain, bounded exactly like the Streamlit sliders."""

    domain: str
    sample_size: int = config.DEFAULT_MOTOR_SAMPLE_SIZE
    inpatient_sample_size: int = config.DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE
    outpatient_sample_size: int = config.DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE

    def cache_key(self) -> tuple:
        if self.domain == MOTOR_DOMAIN:
            return (MOTOR_DOMAIN, self.sample_size)
        return (HEALTH_DOMAIN, self.inpatient_sample_size, self.outpatient_sample_size)

    def describe(self) -> dict[str, int]:
        if self.domain == MOTOR_DOMAIN:
            return {"sample_size": self.sample_size}
        return {
            "inpatient_sample_size": self.inpatient_sample_size,
            "outpatient_sample_size": self.outpatient_sample_size,
        }


@dataclass
class ScoredSample:
    """A sampled dataset plus its `predict_proba` scores, investigation table
    and candidate rings."""

    params: SampleParams
    dataset: Dataset
    table: pd.DataFrame
    rings: RingDetectionResult


@dataclass
class ModelSlot:
    model: Optional[Model]
    directory: Path
    error: Optional[str] = None


@dataclass
class AppState:
    models: dict[str, ModelSlot] = field(default_factory=dict)
    reviews: ReviewStore = field(default_factory=lambda: ReviewStore(config.reviews_path()))
    _samples: "OrderedDict[tuple, ScoredSample]" = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _build_locks: dict[tuple, threading.Lock] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "AppState":
        """Load both persisted models from disk; a missing artifact is recorded, not raised."""
        state = cls()
        for domain, directory, loader in (
            (MOTOR_DOMAIN, config.motor_model_dir(), load_motor_model_or_raise),
            (HEALTH_DOMAIN, config.health_model_dir(), load_health_model_or_raise),
        ):
            try:
                state.models[domain] = ModelSlot(model=loader(directory), directory=directory)
            except ModelNotAvailableError as exc:
                state.models[domain] = ModelSlot(model=None, directory=directory, error=str(exc))
        return state

    def model(self, domain: str) -> Model:
        slot = self.models.get(domain)
        if slot is None or slot.model is None:
            raise ModelUnavailable(slot.error if slot else f"No model slot for domain {domain!r}")
        return slot.model

    def sample(self, params: SampleParams) -> ScoredSample:
        """Build (or reuse) the scored sample for `params`.

        Concurrent requests for the same key wait for one build rather than
        racing to build it twice.
        """
        key = params.cache_key()
        with self._lock:
            cached = self._samples.get(key)
            if cached is not None:
                self._samples.move_to_end(key)
                return cached
            build_lock = self._build_locks.setdefault(key, threading.Lock())

        with build_lock:
            with self._lock:
                cached = self._samples.get(key)
                if cached is not None:
                    return cached
            model = self.model(params.domain)
            dataset = _build_dataset(params)
            scores = score_loaded_sample(model, dataset.X, DEFAULT_RISK_THRESHOLDS)
            table = build_investigation_table(params.domain, dataset, scores)
            rings = detect_candidate_rings(
                claim_entities_from_graph(dataset.graph, NETWORK_ENTITY_TYPES[params.domain]),
                NETWORK_DETECTOR,
                n_claims=len(dataset.X),
            )
            scored = ScoredSample(params=params, dataset=dataset, table=table, rings=rings)
            with self._lock:
                self._samples[key] = scored
                while len(self._samples) > MAX_CACHED_SAMPLES:
                    self._samples.popitem(last=False)
            return scored


def _build_dataset(params: SampleParams) -> Dataset:
    try:
        if params.domain == MOTOR_DOMAIN:
            return data_loader.build_motor_dataset(params.sample_size, config.SAMPLE_SEED)
        return data_loader.build_health_dataset(
            params.inpatient_sample_size, params.outpatient_sample_size, config.SAMPLE_SEED
        )
    except data_loader.DatasetNotAvailableError as exc:
        raise DataUnavailable(str(exc)) from exc
