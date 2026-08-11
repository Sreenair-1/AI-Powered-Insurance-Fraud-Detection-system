"""Configuration for the Streamlit app: artifact paths and interactive sample sizes.

Model artifact directories default to the already-gitignored top-level
`models/` directory (see `.gitignore`, `docs/model_persistence.md`) and can
be overridden per-process via environment variables, so a deployment can
point at a specific trained artifact without editing code.

Sample sizes exist because `src/graph/features.py`'s betweenness centrality
is an exact O(V*E) computation that does not scale to the full raw dataset
sizes (15,420 motor claims / ~558k health claims) within an interactive
session -- verified empirically before picking these defaults (see
`docs/app_ui.md`). The app therefore works over a bounded, seeded sample by
default; the training script (`scripts/train_models.py`) exposes the same
knob for a larger, offline training run.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.data.schemas import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_HEALTH_MODEL_DIR = MODELS_DIR / "health"
DEFAULT_MOTOR_MODEL_DIR = MODELS_DIR / "motor"

HEALTH_MODEL_DIR_ENV = "FRAUD_APP_HEALTH_MODEL_DIR"
MOTOR_MODEL_DIR_ENV = "FRAUD_APP_MOTOR_MODEL_DIR"


def health_model_dir() -> Path:
    """Directory to load the health model artifact from (env-overridable)."""
    return Path(os.environ.get(HEALTH_MODEL_DIR_ENV, str(DEFAULT_HEALTH_MODEL_DIR)))


def motor_model_dir() -> Path:
    """Directory to load the motor model artifact from (env-overridable)."""
    return Path(os.environ.get(MOTOR_MODEL_DIR_ENV, str(DEFAULT_MOTOR_MODEL_DIR)))


# Seeded, bounded samples for the interactive app (see module docstring).
# Empirically: 500 motor claims ~2.3s to build (incl. graph features), 800
# ~5.6s, 2000 ~46s on the reference dev machine -- kept small so claim
# selection stays responsive.
DEFAULT_MOTOR_SAMPLE_SIZE = 500
DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE = 150
DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE = 350
SAMPLE_SEED = 42

MAX_MOTOR_SAMPLE_SIZE = 3000
MAX_HEALTH_INPATIENT_SAMPLE_SIZE = 1200
MAX_HEALTH_OUTPATIENT_SAMPLE_SIZE = 2800

DEFAULT_TOP_N_FACTORS = 10
