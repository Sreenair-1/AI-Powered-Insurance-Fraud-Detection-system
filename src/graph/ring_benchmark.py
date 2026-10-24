"""Synthetic fraud-ring benchmark for `src.graph.rings`.

No labeled Indian claims data records which claims belong to a ring, so the
ring detector is measured on generated claim populations where the ground
truth is known by construction. The generator aims to be harder than the
seeded augmentation in `src.data.synthetic` (which every ring member shares
one exclusive garage):

- Garages are shared legitimately and unevenly (a few very busy ones).
- Bank accounts and phone numbers are mostly unique per claimant, but
  households share them, a legitimate source of false alarms.
- Ring members share a garage (often a busy legitimate one), a mule bank
  account and an agent's phone number, but each member uses each shared
  entity only with some probability (partial overlap).
- Ring sizes vary.
- With probability `noise`, any recorded value is reformatted (case, spaces,
  hyphens) or carries a one-character typo, the way details drift between
  claim forms.

Metrics are claim-level precision/recall of ring membership and ring-level
recovery (a ring counts as recovered when one candidate ring overlaps it with
Jaccard similarity of at least 0.5).

Limitations: the generator encodes assumptions about how rings behave; real
rings that rotate accounts, use many garages or spread across insurers will
look different, so these results bound the detector on these patterns only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.graph.rings import (
    CLAIM_KEY_COL,
    DEFAULT_DETECTOR,
    ENTITY_TYPE_COL,
    ENTITY_VALUE_COL,
    MATCHING_MODES,
    RingDetectorConfig,
    detect_candidate_rings,
    normalize_entity_value,
)

BENCHMARK_VERSION = 1
RANDOM_STATE = 42
ENTITY_TYPES = ("garage", "bank_account", "phone")
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)
ALPHA_LEVELS = (1e-2, 1e-3, 1e-4, 1e-6)
RING_SIZE_LEVELS = (3, 4, 5)
# Claimants per household sharing one bank account and phone (joint families
# reach 4-5), so no ring-size cut-off removes every household.
HOUSEHOLD_SIZES = (2, 3, 4, 5)
HOUSEHOLD_SIZE_WEIGHTS = (0.5, 0.25, 0.15, 0.10)
RECOVERY_JACCARD = 0.5

_SURNAMES = ("Sharma", "Patil", "Verma", "Kulkarni", "Singh", "Desai", "Gupta", "Joshi", "Yadav", "Iyer",
             "Reddy", "Naidu", "Khan", "Shaikh", "Pawar", "Jadhav", "Mehta", "Shah", "Chauhan", "Mishra",
             "Pandey", "Tiwari", "Nair", "Menon", "Rao", "Bose", "Ghosh", "Das", "Sinha", "Thakur")
_GARAGE_SUFFIXES = ("Auto Works", "Motors", "Car Care", "Automobiles", "Service Centre", "Garage",
                    "Auto Clinic", "Wheels")
_CITIES = ("Pune", "Nagpur", "Indore", "Lucknow", "Jaipur", "Nashik", "Surat", "Bhopal", "Kanpur", "Thane",
           "Vadodara", "Ludhiana", "Patna", "Agra", "Rajkot")


@dataclass(frozen=True)
class RingScenario:
    n_claims: int = 15420  # size of the Kaggle motor dataset this stands in for
    n_rings: int = 40
    min_ring_size: int = 3
    max_ring_size: int = 12
    garage_count: int = 400
    household_rate: float = 0.03  # share of claims that head a household sharing bank + phone
    reuse_busy_garage: float = 0.4  # chance a ring routes through an existing popular garage
    linkage: tuple[tuple[str, float], ...] = (("garage", 0.7), ("bank_account", 0.8), ("phone", 0.6))
    noise: float = 0.0
    seed: int = RANDOM_STATE


@dataclass
class GeneratedPopulation:
    claim_entities: pd.DataFrame
    ring_of: dict[str, str]  # ring members only
    claim_keys: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _digits(rng: np.random.Generator, n: int, first: str = "123456789") -> str:
    return str(rng.choice(list(first))) + "".join(str(d) for d in rng.integers(0, 10, size=n - 1))


def _unique(rng: np.random.Generator, make: Callable[[], str], used: set[str]) -> str:
    while True:
        value = make()
        if value not in used:
            used.add(value)
            return value


def _garage_name(rng: np.random.Generator) -> str:
    return f"{rng.choice(_SURNAMES)} {rng.choice(_GARAGE_SUFFIXES)}, {rng.choice(_CITIES)}"


def _reformat(value: str, rng: np.random.Generator) -> str:
    if value.isdigit():
        step = int(rng.choice([3, 4]))
        sep = str(rng.choice([" ", "-"]))
        return sep.join(value[i:i + step] for i in range(0, len(value), step))
    return str(rng.choice([value.upper(), value.lower(), value.replace(",", ""), value.replace(" ", "  ")]))


def _typo(value: str, rng: np.random.Generator) -> str:
    chars = list(value)
    positions = [i for i, c in enumerate(chars) if c.isalnum()]
    if len(positions) < 2:
        return value
    pos = int(rng.choice(positions[:-1]))
    op = str(rng.choice(["substitute", "transpose", "delete", "insert"]))
    alphabet = "0123456789" if chars[pos].isdigit() else "abcdefghijklmnopqrstuvwxyz"
    if op == "substitute":
        chars[pos] = str(rng.choice([c for c in alphabet if c != chars[pos].lower()]))
    elif op == "transpose":
        nxt = positions[positions.index(pos) + 1]
        chars[pos], chars[nxt] = chars[nxt], chars[pos]
    elif op == "delete":
        del chars[pos]
    else:
        chars.insert(pos, str(rng.choice(list(alphabet))))
    return "".join(chars)


def _drift(value: str, noise: float, rng: np.random.Generator) -> str:
    """With probability `noise`, reformat the value or add a one-character typo."""
    if noise <= 0 or rng.random() >= noise:
        return value
    return _reformat(value, rng) if rng.random() < 0.5 else _typo(value, rng)


def generate_population(scenario: RingScenario) -> GeneratedPopulation:
    rng = np.random.default_rng(scenario.seed)
    n = scenario.n_claims
    claim_keys = [f"claim:{i}" for i in range(n)]

    used_names: set[str] = set()
    garages = [_unique(rng, lambda: _garage_name(rng), used_names) for _ in range(scenario.garage_count)]
    popularity = rng.lognormal(0.0, 1.0, size=len(garages))
    popularity /= popularity.sum()

    used_numbers: set[str] = set()
    record = {
        "garage": [garages[i] for i in rng.choice(len(garages), size=n, p=popularity)],
        "bank_account": [_unique(rng, lambda: _digits(rng, 12), used_numbers) for _ in range(n)],
        "phone": [_unique(rng, lambda: _digits(rng, 10, "6789"), used_numbers) for _ in range(n)],
    }

    order = list(rng.permutation(n))
    ring_of: dict[str, str] = {}
    for ring_idx in range(scenario.n_rings):
        size = int(rng.integers(scenario.min_ring_size, scenario.max_ring_size + 1))
        members = [order.pop() for _ in range(min(size, len(order)))]
        ring_id = f"RING{ring_idx + 1:03d}"
        shared = {
            "garage": (garages[int(rng.choice(len(garages), p=popularity))]
                       if rng.random() < scenario.reuse_busy_garage
                       else _unique(rng, lambda: _garage_name(rng), used_names)),
            "bank_account": _unique(rng, lambda: _digits(rng, 12), used_numbers),
            "phone": _unique(rng, lambda: _digits(rng, 10, "6789"), used_numbers),
        }
        linkage = dict(scenario.linkage)
        for member in members:
            uses = [t for t in ENTITY_TYPES if rng.random() < linkage[t]]
            if not uses:  # every member shares at least one ring entity
                uses = [str(rng.choice(ENTITY_TYPES))]
            for entity_type in uses:
                record[entity_type][member] = shared[entity_type]
            ring_of[claim_keys[member]] = ring_id

    for _ in range(int(n * scenario.household_rate)):
        size = int(rng.choice(HOUSEHOLD_SIZES, p=HOUSEHOLD_SIZE_WEIGHTS))
        if len(order) < size:
            break
        head, *family = [order.pop() for _ in range(size)]
        for member in family:
            record["bank_account"][member] = record["bank_account"][head]
            record["phone"][member] = record["phone"][head]

    rows = [
        (claim_keys[i], entity_type, _drift(record[entity_type][i], scenario.noise, rng))
        for i in range(n)
        for entity_type in ENTITY_TYPES
    ]
    table = pd.DataFrame(rows, columns=[CLAIM_KEY_COL, ENTITY_TYPE_COL, ENTITY_VALUE_COL])
    return GeneratedPopulation(claim_entities=table, ring_of=ring_of, claim_keys=claim_keys)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _prf(flagged: set[str], positives: set[str]) -> dict[str, Optional[float]]:
    tp = len(flagged & positives)
    precision = tp / len(flagged) if flagged else None
    recall = tp / len(positives) if positives else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_flagged": len(flagged)}


def evaluate_detection(population: GeneratedPopulation, config: RingDetectorConfig) -> dict[str, Any]:
    result = detect_candidate_rings(population.claim_entities, config, n_claims=len(population.claim_keys))
    positives = set(population.ring_of)
    metrics = _prf(result.flagged_claims(), positives)

    true_rings: dict[str, set[str]] = {}
    for claim, ring_id in population.ring_of.items():
        true_rings.setdefault(ring_id, set()).add(claim)
    candidates = [set(r.claims) for r in result.rings]
    recovered = sum(
        any(len(t & c) / len(t | c) >= RECOVERY_JACCARD for c in candidates) for t in true_rings.values()
    )
    false_rings = sum(len(c & positives) * 2 < len(c) for c in candidates)
    metrics.update({
        "ring_recovery_rate": recovered / len(true_rings) if true_rings else None,
        "n_candidate_rings": len(candidates),
        "n_false_candidate_rings": false_rings,
    })
    return metrics


def evaluate_naive_baseline(population: GeneratedPopulation) -> dict[str, Any]:
    """Flag any claim sharing any (normalised) entity value with another claim —
    the existing `has_shared_entity` graph feature used as a detector."""
    table = population.claim_entities.assign(
        key=lambda t: t[ENTITY_TYPE_COL] + ":" + t[ENTITY_VALUE_COL].map(normalize_entity_value)
    )
    degree = table.groupby("key")[CLAIM_KEY_COL].transform("nunique")
    flagged = set(table.loc[degree >= 2, CLAIM_KEY_COL])
    return _prf(flagged, set(population.ring_of))


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in runs[0]:
        values = [r[key] for r in runs if r[key] is not None]
        if not values:
            out[key] = None
            continue
        out[key] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    return out


DEFAULT_SCENARIO = RingScenario()


def run_ring_benchmark(
    scenario: RingScenario = DEFAULT_SCENARIO,
    seeds: tuple[int, ...] = (42, 43, 44),
    noise_levels: tuple[float, ...] = NOISE_LEVELS,
    alpha_levels: tuple[float, ...] = ALPHA_LEVELS,
    ring_size_levels: tuple[int, ...] = RING_SIZE_LEVELS,
    default_config: RingDetectorConfig = DEFAULT_DETECTOR,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Noise sweep (every matching mode) plus an alpha x minimum-ring-size
    operating-point sweep, averaged over seeds."""
    noise_sweep = []
    for noise in noise_levels:
        populations = [generate_population(replace(scenario, noise=noise, seed=s)) for s in seeds]
        row: dict[str, Any] = {"noise": noise,
                               "naive_any_shared_entity": _aggregate([evaluate_naive_baseline(p) for p in populations])}
        for matching in MATCHING_MODES:
            config = replace(default_config, matching=matching)
            row[matching] = _aggregate([evaluate_detection(p, config) for p in populations])
        noise_sweep.append(row)
        if progress is not None:
            progress(f"noise {noise:.1f} done")

    sweep_noise = 0.2 if 0.2 in noise_levels else noise_levels[len(noise_levels) // 2]
    populations = [generate_population(replace(scenario, noise=sweep_noise, seed=s)) for s in seeds]
    operating_points = [
        {"alpha": alpha, "min_ring_size": size, **_aggregate([
            evaluate_detection(p, replace(default_config, matching="fuzzy", alpha=alpha, min_ring_size=size))
            for p in populations
        ])}
        for alpha in alpha_levels
        for size in ring_size_levels
    ]

    sample = generate_population(replace(scenario, seed=seeds[0]))
    return {
        "benchmark": "fraud_ring_recovery",
        "version": BENCHMARK_VERSION,
        "seeds": list(seeds),
        "scenario": {k: v for k, v in asdict(scenario).items() if k not in {"noise", "seed"}},
        "ring_member_prevalence": len(sample.ring_of) / scenario.n_claims,
        "detector": {
            "method": "unexpected co-sharing of entity pairs (Poisson tail test), connected into candidate rings",
            "alpha": default_config.alpha,
            "min_ring_size": default_config.min_ring_size,
            "matching_modes": {
                "raw": "values must match character for character",
                "normalized": "case, spaces and punctuation ignored",
                "fuzzy": "normalised values within one edit (typo) are merged",
            },
        },
        "recovery_rule": f"a true ring is recovered when one candidate ring has Jaccard >= {RECOVERY_JACCARD} with it",
        "noise_sweep": noise_sweep,
        "operating_points": {"noise": sweep_noise, "matching": "fuzzy", "rows": operating_points},
        "limitations": [
            "Claim populations and rings are generated; real rings may rotate accounts or spread across insurers",
            "Households share bank accounts and phones legitimately, so some candidate rings are families",
            "Entities are garage, bank account and phone (motor-style); doctors play the garage role for health",
        ],
    }
