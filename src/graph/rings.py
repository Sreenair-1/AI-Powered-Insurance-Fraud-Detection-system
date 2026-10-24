"""Candidate fraud-ring detection from shared claim entities.

Sharing one entity is normal: a busy garage or hospital serves thousands of
honest claims. What a ring leaves behind is *unexpected co-sharing*: several
claims that use the same pair of entities (e.g. one garage and one bank
account) far more often than those entities' own popularity explains.

For every pair of entities of different types seen together on `c >= 2`
claims, the chance expectation under independent assignment is
`expected = deg(e1) * deg(e2) / n_claims`, and the pair is significant when
the Poisson tail `P(X >= c | expected)` is at most `alpha`. Claims joined by a
significant pair are linked; connected groups of at least `min_ring_size`
claims are candidate rings.

Entity values can be matched exactly, after normalisation (case, spaces and
punctuation removed), or with typo tolerance (normalised values within one
edit of each other are merged), so the same garage written two ways still
links its claims.

Blind spot: evidence always involves two entity types, so claims that share
only one entity (e.g. one bank account, at garages everyone uses) are not
flagged, and households sharing an account and phone can be.

This is decision support: a candidate ring is a pattern worth an
investigator's time, never proof of collusion.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Hashable, Iterable, Optional

import networkx as nx
import pandas as pd
from scipy.stats import poisson

from src.graph.construction import CLAIM_NODE_TYPE, NODE_TYPE_ATTR

MATCHING_MODES = ("raw", "normalized", "fuzzy")
# Typo tolerance is only applied to values this long; short codes differ by
# one character far too often to merge safely.
FUZZY_MIN_LENGTH = 6

CLAIM_KEY_COL = "claim_key"
ENTITY_TYPE_COL = "entity_type"
ENTITY_VALUE_COL = "entity_value"


@dataclass(frozen=True)
class RingDetectorConfig:
    alpha: float = 1e-3
    min_ring_size: int = 3
    matching: str = "normalized"

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.min_ring_size < 2:
            raise ValueError(f"min_ring_size must be >= 2, got {self.min_ring_size}")
        if self.matching not in MATCHING_MODES:
            raise ValueError(f"matching must be one of {MATCHING_MODES}, got {self.matching!r}")


@dataclass(frozen=True)
class SharedEntityPair:
    """One significant co-sharing pattern: `claims` all use both entities."""

    entities: tuple[str, str]
    claims: tuple[Hashable, ...]
    expected: float
    p_value: float


@dataclass(frozen=True)
class CandidateRing:
    ring_id: str
    claims: tuple[Hashable, ...]
    evidence: tuple[SharedEntityPair, ...]


@dataclass
class RingDetectionResult:
    config: RingDetectorConfig
    n_claims: int
    rings: list[CandidateRing]
    # Every claim joined by a significant pair, including pairs too small to be a ring.
    linked_claims: set[Hashable] = field(default_factory=set)

    def ring_of(self) -> dict[Hashable, CandidateRing]:
        return {claim: ring for ring in self.rings for claim in ring.claims}

    def flagged_claims(self) -> set[Hashable]:
        return {claim for ring in self.rings for claim in ring.claims}


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------


def normalize_entity_value(value: Any) -> str:
    """Casefold and drop everything but letters and digits."""
    return re.sub(r"[^0-9a-z]", "", str(value).casefold())


def _within_one_edit(a: str, b: str) -> bool:
    """Optimal-string-alignment distance <= 1 (substitution, insertion,
    deletion, or one adjacent transposition)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    # b is one longer than a: removing one character of b must give a.
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def resolve_entity_values(values: Iterable[Any], matching: str) -> dict[Any, str]:
    """Map each raw value to a canonical key according to `matching`."""
    values = list(dict.fromkeys(values))
    if matching == "raw":
        return {v: str(v) for v in values}
    normalized = {v: normalize_entity_value(v) for v in values}
    if matching == "normalized":
        return normalized

    # Typo-tolerant: union normalised values within one edit. Candidate pairs
    # come from shared one-deletion variants (symmetric-delete blocking), so
    # the cost stays linear in the number of values rather than quadratic.
    distinct = sorted(set(normalized.values()))
    parent = {s: s for s in distinct}

    def find(s: str) -> str:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    buckets: dict[str, list[str]] = defaultdict(list)
    for s in distinct:
        if len(s) < FUZZY_MIN_LENGTH:
            continue
        buckets[s].append(s)
        for i in range(len(s)):
            buckets[s[:i] + s[i + 1:]].append(s)
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(set(members)), 2):
            if _within_one_edit(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return {v: find(n) for v, n in normalized.items()}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


DEFAULT_DETECTOR = RingDetectorConfig()


def detect_candidate_rings(
    claim_entities: pd.DataFrame,
    config: RingDetectorConfig = DEFAULT_DETECTOR,
    n_claims: Optional[int] = None,
) -> RingDetectionResult:
    """Detect candidate rings from a long table of `claim_key, entity_type, entity_value`.

    `n_claims` defaults to the number of distinct claims in the table; pass
    the full population size if some claims have no entities listed.
    """
    missing = {CLAIM_KEY_COL, ENTITY_TYPE_COL, ENTITY_VALUE_COL} - set(claim_entities.columns)
    if missing:
        raise ValueError(f"claim_entities is missing columns {sorted(missing)}")
    table = claim_entities.dropna(subset=[ENTITY_VALUE_COL])
    n = int(n_claims if n_claims is not None else table[CLAIM_KEY_COL].nunique())
    if n == 0 or table.empty:
        return RingDetectionResult(config=config, n_claims=n, rings=[])

    resolved = []
    for entity_type, group in table.groupby(ENTITY_TYPE_COL, sort=True):
        mapping = resolve_entity_values(group[ENTITY_VALUE_COL], config.matching)
        resolved.append(pd.DataFrame({
            CLAIM_KEY_COL: group[CLAIM_KEY_COL].to_numpy(),
            "entity": [f"{entity_type}:{mapping[v]}" for v in group[ENTITY_VALUE_COL]],
            ENTITY_TYPE_COL: entity_type,
        }))
    entities = pd.concat(resolved, ignore_index=True).drop_duplicates([CLAIM_KEY_COL, "entity"])
    degree = entities.groupby("entity")[CLAIM_KEY_COL].nunique()
    entity_type_of = dict(zip(entities["entity"], entities[ENTITY_TYPE_COL], strict=True))

    pair_claims: dict[tuple[str, str], list[Hashable]] = defaultdict(list)
    for claim, claim_group in entities.groupby(CLAIM_KEY_COL, sort=False):
        shared = sorted(e for e in claim_group["entity"] if degree[e] >= 2)
        for e1, e2 in combinations(shared, 2):
            if entity_type_of[e1] != entity_type_of[e2]:
                pair_claims[(e1, e2)].append(claim)

    linker = nx.Graph()
    significant: list[SharedEntityPair] = []
    for (e1, e2), claims in pair_claims.items():
        count = len(claims)
        if count < 2:
            continue
        expected = float(degree[e1] * degree[e2] / n)
        p_value = float(poisson.sf(count - 1, expected))
        if p_value > config.alpha:
            continue
        pattern = SharedEntityPair((e1, e2), tuple(claims), expected, p_value)
        significant.append(pattern)
        linker.add_edges_from((claims[0], other) for other in claims[1:])

    by_claim: dict[Hashable, list[SharedEntityPair]] = defaultdict(list)
    for pattern in significant:
        for claim in pattern.claims:
            by_claim[claim].append(pattern)

    components = sorted(
        (sorted(c, key=str) for c in nx.connected_components(linker) if len(c) >= config.min_ring_size),
        key=lambda members: (-len(members), str(members[0])),
    )
    rings = []
    for idx, members in enumerate(components):
        evidence = {id(p): p for claim in members for p in by_claim[claim]}
        ordered = sorted(evidence.values(), key=lambda p: (p.p_value, -len(p.claims)))
        rings.append(CandidateRing(ring_id=f"CR{idx + 1:03d}", claims=tuple(members), evidence=tuple(ordered)))
    return RingDetectionResult(config=config, n_claims=n, rings=rings, linked_claims=set(linker.nodes))


def claim_entities_from_graph(graph: nx.Graph, entity_types: Iterable[str]) -> pd.DataFrame:
    """Long `claim_key, entity_type, entity_value` table from an entity graph.

    Node ids are already type-prefixed and exact, so the value is the raw id
    after the prefix.
    """
    wanted = set(entity_types)
    rows = []
    for node, attrs in graph.nodes(data=True):
        if attrs.get(NODE_TYPE_ATTR) != CLAIM_NODE_TYPE:
            continue
        for neighbor in graph.neighbors(node):
            kind = graph.nodes[neighbor].get(NODE_TYPE_ATTR)
            if kind in wanted:
                rows.append((node, kind, str(neighbor).split(":", 1)[-1]))
    return pd.DataFrame(rows, columns=[CLAIM_KEY_COL, ENTITY_TYPE_COL, ENTITY_VALUE_COL])


def detection_to_frame(result: RingDetectionResult, claim_keys: Iterable[Hashable]) -> pd.DataFrame:
    """One row per claim: candidate ring id/size and whether it has any significant link."""
    ring_of = result.ring_of()
    rows = []
    for claim in claim_keys:
        ring = ring_of.get(claim)
        rows.append({
            CLAIM_KEY_COL: claim,
            "candidate_ring_id": ring.ring_id if ring else None,
            "candidate_ring_size": len(ring.claims) if ring else 0,
            "has_significant_link": claim in result.linked_claims,
        })
    return pd.DataFrame(rows)
