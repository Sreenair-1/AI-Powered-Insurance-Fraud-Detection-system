"""Unit tests for src/graph/rings.py and src/graph/ring_benchmark.py."""
from dataclasses import replace

import pandas as pd
import pytest

from src.data.synthetic import FraudRingConfig, assign_synthetic_entities, inject_fraud_rings
from src.graph.construction import build_motor_graph
from src.graph.ring_benchmark import (
    RingScenario,
    evaluate_detection,
    evaluate_naive_baseline,
    generate_population,
    run_ring_benchmark,
)
from src.graph.rings import (
    RingDetectorConfig,
    _within_one_edit,
    claim_entities_from_graph,
    detect_candidate_rings,
    normalize_entity_value,
    resolve_entity_values,
)


def _table(rows):
    return pd.DataFrame(rows, columns=["claim_key", "entity_type", "entity_value"])


def test_normalize_drops_case_spaces_and_punctuation():
    assert normalize_entity_value("Sharma Motors, Pune") == normalize_entity_value("SHARMA  MOTORS PUNE")
    assert normalize_entity_value("1234-5678 9012") == "123456789012"


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("123456789012", "123456789012", True),
        ("123456789012", "123456789013", True),  # substitution
        ("123456789012", "123456798012", True),  # adjacent transposition
        ("123456789012", "12345679012", True),  # deletion
        ("123456789012", "1234567890125", True),  # insertion
        ("123456789012", "123456780000", False),
        ("123456789012", "1234567890", False),
    ],
)
def test_within_one_edit(a, b, expected):
    assert _within_one_edit(a, b) is expected
    assert _within_one_edit(b, a) is expected


def test_fuzzy_resolution_merges_typos_but_not_distinct_values():
    values = ["Sharma Motors, Pune", "sharma motor pune", "Patil Auto Works, Nashik", "623145559012"]
    fuzzy = resolve_entity_values(values, "fuzzy")
    assert fuzzy["Sharma Motors, Pune"] == fuzzy["sharma motor pune"]
    assert len({fuzzy[v] for v in values}) == 3
    normalized = resolve_entity_values(values, "normalized")
    assert normalized["Sharma Motors, Pune"] != normalized["sharma motor pune"]
    raw = resolve_entity_values(["A-1", "a1"], "raw")
    assert raw["A-1"] != raw["a1"]


def test_fuzzy_never_merges_short_codes():
    mapping = resolve_entity_values(["G1", "G2", "G3"], "fuzzy")
    assert len(set(mapping.values())) == 3


def _population_with_ring(n_claims: int = 300, n_garages: int = 30):
    """Claims spread over busy garages with their own accounts; claims 0-2 use
    garage 0 *and* one shared mule account."""
    rows = []
    for i in range(n_claims):
        rows.append((f"claim:{i}", "garage", f"Garage {i % n_garages:03d}"))
        account = "MULE000001" if i in (0, 30, 60) else f"ACC{i:06d}"
        rows.append((f"claim:{i}", "bank_account", account))
    return _table(rows)


def test_detects_unexpected_co_sharing_and_ignores_busy_garages():
    result = detect_candidate_rings(_population_with_ring(), RingDetectorConfig(min_ring_size=3))
    assert len(result.rings) == 1
    ring = result.rings[0]
    assert set(ring.claims) == {"claim:0", "claim:30", "claim:60"}
    evidence = ring.evidence[0]
    assert set(evidence.entities) == {"garage:garage000", "bank_account:mule000001"}
    assert evidence.expected == pytest.approx(10 * 3 / 300)
    # The other claims sharing only a busy garage are not flagged.
    assert result.flagged_claims() == set(ring.claims)


def test_one_shared_entity_alone_is_not_evidence():
    """Documented blind spot: the test needs two entity types co-occurring, so
    a mule account shared by claims at a garage everyone uses is not flagged."""
    rows = [(f"claim:{i}", "garage", "Busy Garage") for i in range(60)]
    rows += [(f"claim:{i}", "bank_account", "MULE" if i < 3 else f"ACC{i:06d}") for i in range(60)]
    assert detect_candidate_rings(_table(rows), RingDetectorConfig()).rings == []


def test_sharing_a_busy_hub_and_one_rare_entity_is_not_significant_when_expected():
    """If the 'rare' entity is itself as busy as the hub, co-sharing is expected."""
    rows = [(f"claim:{i}", t, v) for i in range(40) for t, v in (("garage", "G-ALPHA"), ("bank_account", "B-ALPHA"))]
    result = detect_candidate_rings(_table(rows), RingDetectorConfig())
    assert result.rings == []


def test_min_ring_size_separates_pairs_from_rings():
    rows = [("c1", "garage", "GARAGE-X"), ("c1", "bank_account", "ACC-X"),
            ("c2", "garage", "GARAGE-X"), ("c2", "bank_account", "ACC-X")]
    rows += [(f"o{i}", "garage", f"GARAGE-{i:04d}") for i in range(200)]
    result = detect_candidate_rings(_table(rows), RingDetectorConfig(min_ring_size=3))
    assert result.rings == []
    assert result.linked_claims == {"c1", "c2"}


def test_detector_config_validation():
    with pytest.raises(ValueError):
        RingDetectorConfig(alpha=0)
    with pytest.raises(ValueError):
        RingDetectorConfig(min_ring_size=1)
    with pytest.raises(ValueError):
        RingDetectorConfig(matching="phonetic")
    with pytest.raises(ValueError):
        detect_candidate_rings(pd.DataFrame({"claim_key": []}))


def test_empty_input_returns_no_rings():
    assert detect_candidate_rings(_table([])).rings == []


def test_claim_entities_from_motor_graph_and_injected_rings_are_found():
    claims = pd.DataFrame({"x": range(400)})
    with_entities = assign_synthetic_entities(claims, seed=3)
    augmented, assignments = inject_fraud_rings(
        with_entities, FraudRingConfig(num_rings=3, ring_size=5, linkage_strength=1.0, seed=3)
    )
    graph = build_motor_graph(augmented)
    table = claim_entities_from_graph(graph, ["garage", "bank_account"])
    assert len(table) == 2 * len(augmented)
    assert set(table["entity_type"]) == {"garage", "bank_account"}

    result = detect_candidate_rings(table, RingDetectorConfig(matching="raw"), n_claims=len(augmented))
    truth = {f"claim:{i}" for i in assignments["claim_row_index"]}
    flagged = result.flagged_claims()
    assert truth <= flagged
    # The seeded augmentation's small entity pools make some honest claims
    # coincide on both garage and account, so a few extra flags are expected.
    assert len(truth) / len(flagged) >= 0.5


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

SMALL = RingScenario(n_claims=1500, n_rings=8, garage_count=60, seed=5)


def test_generated_population_has_known_rings_and_unique_accounts():
    population = generate_population(SMALL)
    table = population.claim_entities
    assert len(table) == 3 * SMALL.n_claims
    assert 8 * SMALL.min_ring_size <= len(population.ring_of) <= 8 * SMALL.max_ring_size
    banks = table[table["entity_type"] == "bank_account"]["entity_value"]
    # Mostly unique accounts: households and rings are the only sharing.
    assert banks.nunique() > 0.85 * SMALL.n_claims
    assert generate_population(SMALL).claim_entities.equals(table)  # deterministic


def test_noise_drifts_values_and_typo_tolerance_recovers_more_rings():
    clean = generate_population(SMALL)
    noisy = generate_population(replace(SMALL, noise=0.4))
    changed = (clean.claim_entities["entity_value"] != noisy.claim_entities["entity_value"]).mean()
    assert 0.2 < changed < 0.6

    raw = evaluate_detection(noisy, RingDetectorConfig(matching="raw"))
    fuzzy = evaluate_detection(noisy, RingDetectorConfig(matching="fuzzy"))
    assert fuzzy["ring_recovery_rate"] > raw["ring_recovery_rate"]
    assert fuzzy["recall"] > raw["recall"]


def test_detector_beats_naive_any_shared_entity_rule():
    population = generate_population(SMALL)
    naive = evaluate_naive_baseline(population)
    detected = evaluate_detection(population, RingDetectorConfig(matching="fuzzy"))
    assert naive["recall"] > 0.9
    assert detected["precision"] > 3 * naive["precision"]


def test_run_ring_benchmark_report_shape():
    report = run_ring_benchmark(
        replace(SMALL, n_claims=800, n_rings=4, garage_count=40),
        seeds=(1,), noise_levels=(0.0, 0.3), alpha_levels=(1e-3,), ring_size_levels=(3, 4),
    )
    assert [row["noise"] for row in report["noise_sweep"]] == [0.0, 0.3]
    assert set(report["noise_sweep"][0]) == {"noise", "naive_any_shared_entity", "raw", "normalized", "fuzzy"}
    assert len(report["operating_points"]["rows"]) == 2
    assert report["ring_member_prevalence"] > 0
    assert report["limitations"]
