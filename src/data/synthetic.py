"""Synthetic augmentation: Indian-context demographic sampling, synthetic
GarageID/BankAccountID entities, and controlled fraud-ring injection.

This module never touches `data/raw/` and never modifies `FraudFound_P` (the
real motor label). Everything it produces is clearly synthetic:

- `GarageID` / `BankAccountID` do not exist in the real motor dataset — they
  are invented here purely to give the later graph-analytics stage entities
  to link claims through.
- `synthetic_fraud_ring_label` / `synthetic_ring_id` are synthetic ground
  truth for controlled experimentation, never a substitute for real fraud
  evidence — see docs/synthetic_augmentation.md.
- The Indian demographics reference dataset is used only as a reference
  distribution to bootstrap-sample synthetic profiles from; it is never
  merged row-for-row with claims (see docs/data_contract.md section 12).

All randomness is seeded (`numpy.random.RandomState`), default seed 42, so
every function here is deterministic given the same inputs and seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Hashable, cast

from src.data.loaders import load_demographics_reference, load_motor_claims
from src.data.schemas import DATA_GENERATED_DIR

RANDOM_STATE = 42

GARAGE_ID_COL = "GarageID"
BANK_ACCOUNT_ID_COL = "BankAccountID"
RING_LABEL_COL = "synthetic_fraud_ring_label"
RING_ID_COL = "synthetic_ring_id"

DEMOGRAPHIC_PROFILE_PREFIX = "synthetic_demographic_"


# ---------------------------------------------------------------------------
# 1. Indian-context demographic sampling (reference distribution only)
# ---------------------------------------------------------------------------


def sample_indian_demographics(
    n: int,
    seed: int = RANDOM_STATE,
    demographics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Bootstrap-sample `n` synthetic demographic profiles from the Indian reference dataset.

    This is a reference-distribution sample, not a join: rows are drawn with
    replacement from `data/raw/demographics/Indian_Insurance_Data.csv` (or a
    caller-supplied frame for testing) and every column is renamed with a
    `synthetic_demographic_` prefix so it can never be mistaken for a real
    claimant's recorded demographic data. Never merge this output onto real
    claims by row position — it has no claimant identity of its own.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    if demographics_df is None:
        demographics_df = load_demographics_reference()
    if len(demographics_df) == 0:
        raise ValueError("demographics_df is empty; cannot sample a reference distribution")

    rng = np.random.RandomState(seed)
    positions = rng.randint(0, len(demographics_df), size=n)
    sample = demographics_df.iloc[positions].reset_index(drop=True).copy()
    sample.columns = [f"{DEMOGRAPHIC_PROFILE_PREFIX}{c}" for c in sample.columns]
    sample.insert(0, "synthetic_profile_id", [f"PROF{i + 1:06d}" for i in range(n)])
    return sample


# ---------------------------------------------------------------------------
# 2 & 3. Synthetic entity ID generation
# ---------------------------------------------------------------------------


def generate_garage_ids(n: int, prefix: str = "GRG") -> list[str]:
    """Generate `n` unique, deterministic synthetic GarageID strings."""
    return _generate_ids(n, prefix)


def generate_bank_account_ids(n: int, prefix: str = "BNK") -> list[str]:
    """Generate `n` unique, deterministic synthetic BankAccountID strings."""
    return _generate_ids(n, prefix)


def _generate_ids(n: int, prefix: str) -> list[str]:
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    width = max(6, len(str(n)))
    return [f"{prefix}{i + 1:0{width}d}" for i in range(n)]


def assign_synthetic_entities(
    claims_df: pd.DataFrame,
    seed: int = RANDOM_STATE,
    garage_pool_ratio: float = 0.05,
    bank_pool_ratio: float = 0.08,
) -> pd.DataFrame:
    """Attach baseline synthetic `GarageID`/`BankAccountID` columns to every claim.

    Each claim is independently assigned one garage and one bank account
    drawn uniformly from a fixed-size pool (pool sizes derived from
    `garage_pool_ratio`/`bank_pool_ratio`, so many claims naturally share the
    same entity — that alone is not fraud). Returns a copy; `claims_df` is
    never modified. This must run before `inject_fraud_rings`.
    """
    n = len(claims_df)
    if n == 0:
        raise ValueError("claims_df must not be empty")
    if not 0.0 < garage_pool_ratio <= 1.0:
        raise ValueError(f"garage_pool_ratio must be in (0.0, 1.0], got {garage_pool_ratio}")
    if not 0.0 < bank_pool_ratio <= 1.0:
        raise ValueError(f"bank_pool_ratio must be in (0.0, 1.0], got {bank_pool_ratio}")

    num_garages = max(1, round(n * garage_pool_ratio))
    num_banks = max(1, round(n * bank_pool_ratio))
    garage_pool = generate_garage_ids(num_garages)
    bank_pool = generate_bank_account_ids(num_banks)

    rng = np.random.RandomState(seed)
    out = claims_df.copy()
    out[GARAGE_ID_COL] = rng.choice(garage_pool, size=n)
    out[BANK_ACCOUNT_ID_COL] = rng.choice(bank_pool, size=n)
    return out


# ---------------------------------------------------------------------------
# 4-6. Configurable fraud-ring injection with explicit linkage + separate labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FraudRingConfig:
    """Parameters controlling synthetic fraud-ring injection.

    Pattern: `ring_size` claims are selected per ring and all made to share
    one synthetic GarageID; `linkage_strength` is the fraction of those same
    claims that additionally share one synthetic BankAccountID.
    `max_affected_proportion` is a safety rail so this never labels a large
    slice of the dataset as ring members.
    """

    num_rings: int = 5
    ring_size: int = 4
    linkage_strength: float = 0.8
    max_affected_proportion: float = 0.25
    seed: int = RANDOM_STATE

    def __post_init__(self) -> None:
        if self.num_rings < 1:
            raise ValueError(f"num_rings must be >= 1, got {self.num_rings}")
        if self.ring_size < 2:
            raise ValueError(
                f"ring_size must be >= 2 (a ring needs at least 2 claims to share "
                f"entities), got {self.ring_size}"
            )
        if not 0.0 <= self.linkage_strength <= 1.0:
            raise ValueError(f"linkage_strength must be in [0.0, 1.0], got {self.linkage_strength}")
        if not 0.0 < self.max_affected_proportion <= 1.0:
            raise ValueError(
                f"max_affected_proportion must be in (0.0, 1.0], got {self.max_affected_proportion}"
            )


def inject_fraud_rings(
    claims_df: pd.DataFrame,
    config: FraudRingConfig = FraudRingConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject `config.num_rings` synthetic fraud rings into `claims_df`.

    `claims_df` must already have `GarageID`/`BankAccountID` columns (from
    `assign_synthetic_entities`). Returns `(augmented_claims, ring_assignments)`:

    - `augmented_claims` is a copy of `claims_df` with the selected rings'
      `GarageID`/`BankAccountID` overwritten to shared values, plus two new
      columns: `synthetic_fraud_ring_label` (1 for injected ring members, 0
      otherwise) and `synthetic_ring_id` (ring identifier or None). The real
      `FraudFound_P` column, if present, is left untouched.
    - `ring_assignments` is a ground-truth table (one row per ring member)
      recording the explicit claim -> shared-entity linkage, separate from
      any real fraud label.
    """
    missing = {GARAGE_ID_COL, BANK_ACCOUNT_ID_COL} - set(claims_df.columns)
    if missing:
        raise ValueError(
            f"claims_df is missing required columns {sorted(missing)}; call "
            "assign_synthetic_entities() before inject_fraud_rings()."
        )

    n = len(claims_df)
    total_needed = config.num_rings * config.ring_size
    if total_needed > n:
        raise ValueError(
            f"Cannot inject {config.num_rings} rings of size {config.ring_size} into "
            f"{n} claims: needs {total_needed} distinct claims."
        )
    affected_proportion = total_needed / n
    if affected_proportion > config.max_affected_proportion:
        raise ValueError(
            f"Requested rings would affect {affected_proportion:.1%} of claims, "
            f"exceeding max_affected_proportion={config.max_affected_proportion:.1%}. "
            "Reduce num_rings/ring_size or raise max_affected_proportion explicitly."
        )

    rng = np.random.RandomState(config.seed)
    selected_positions = rng.choice(n, size=total_needed, replace=False)

    out = claims_df.copy()
    out[RING_LABEL_COL] = 0
    out[RING_ID_COL] = pd.Series([None] * n, dtype="object", index=out.index)

    ring_rows: list[dict] = []
    for ring_idx in range(config.num_rings):
        ring_id = f"RING{ring_idx + 1:03d}"
        shared_garage = f"GRG-{ring_id}"
        shared_bank = f"BNK-{ring_id}"
        member_positions = selected_positions[
            ring_idx * config.ring_size : (ring_idx + 1) * config.ring_size
        ]
        deep_link_mask = rng.random_sample(config.ring_size) < config.linkage_strength

        for position, deep_link in zip(member_positions, deep_link_mask):
            row_label = cast(Hashable, out.index[position])
            out.at[row_label, GARAGE_ID_COL] = shared_garage
            if deep_link:
                out.at[row_label, BANK_ACCOUNT_ID_COL] = shared_bank
            out.at[row_label, RING_LABEL_COL] = 1
            out.at[row_label, RING_ID_COL] = ring_id
            ring_rows.append(
                {
                    "ring_id": ring_id,
                    "claim_row_index": row_label,
                    "garage_id": shared_garage,
                    "bank_account_id": out.at[row_label, BANK_ACCOUNT_ID_COL],
                    "shares_bank_account": bool(deep_link),
                }
            )

    ring_assignments = pd.DataFrame(
        ring_rows,
        columns=["ring_id", "claim_row_index", "garage_id", "bank_account_id", "shares_bank_account"],
    )
    return out, ring_assignments


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticAugmentationResult:
    """Output of a full synthetic augmentation run. See docs/synthetic_augmentation.md."""

    claims: pd.DataFrame
    ring_assignments: pd.DataFrame
    ring_demographic_profiles: pd.DataFrame


def build_motor_synthetic_augmentation(
    motor_df: pd.DataFrame | None = None,
    ring_config: FraudRingConfig | None = None,
    seed: int = RANDOM_STATE,
    demographics_df: pd.DataFrame | None = None,
    garage_pool_ratio: float = 0.05,
    bank_pool_ratio: float = 0.08,
) -> SyntheticAugmentationResult:
    """Run the full synthetic augmentation pipeline on motor claims.

    Loads `data/raw/motor/fraud_oracle.csv` (or uses `motor_df` if supplied,
    e.g. for tests), assigns baseline synthetic GarageID/BankAccountID
    entities, injects `ring_config` fraud rings, and bootstrap-samples one
    Indian-context demographic profile per ring (kept as a separate table,
    never merged into `claims`). Raw data on disk is never modified.
    """
    if motor_df is None:
        motor_df = load_motor_claims()
    if ring_config is None:
        ring_config = FraudRingConfig(seed=seed)

    with_entities = assign_synthetic_entities(
        motor_df, seed=seed, garage_pool_ratio=garage_pool_ratio, bank_pool_ratio=bank_pool_ratio
    )
    augmented_claims, ring_assignments = inject_fraud_rings(with_entities, config=ring_config)

    ring_profiles = sample_indian_demographics(
        n=ring_config.num_rings, seed=seed, demographics_df=demographics_df
    )
    ring_profiles.insert(
        0, "ring_id", [f"RING{i + 1:03d}" for i in range(ring_config.num_rings)]
    )

    return SyntheticAugmentationResult(
        claims=augmented_claims,
        ring_assignments=ring_assignments,
        ring_demographic_profiles=ring_profiles,
    )


def save_synthetic_augmentation(
    result: SyntheticAugmentationResult, output_dir: Path | None = None
) -> dict[str, Path]:
    """Write a SyntheticAugmentationResult to CSVs under a derived/generated directory.

    Defaults to `data/generated/synthetic/` (never `data/raw/`). Returns the
    written file paths. `data/` is already gitignored except `.gitkeep`, so
    these files are never committed.
    """
    output_dir = output_dir or (DATA_GENERATED_DIR / "synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "claims": output_dir / "motor_claims_synthetic_augmented.csv",
        "ring_assignments": output_dir / "synthetic_ring_assignments.csv",
        "ring_demographic_profiles": output_dir / "synthetic_ring_demographic_profiles.csv",
    }
    result.claims.to_csv(paths["claims"], index=False)
    result.ring_assignments.to_csv(paths["ring_assignments"], index=False)
    result.ring_demographic_profiles.to_csv(paths["ring_demographic_profiles"], index=False)
    return paths
