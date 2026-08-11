"""Graph feature adapter.

Wraps `src.graph.features.extract_claim_features()` and aligns its output
to a known, external set of `claim_key` values (typically the ones
produced by `src.features.structured`), so every structured claim gets a
graph-feature row even when it has no presence in the graph (or the graph
is empty) — instead of silently dropping out of a downstream join.

This module makes no fraud decision; it only reshapes existing graph
features and fills safe defaults for claims absent from the graph.
"""
from __future__ import annotations

from typing import Iterable

import networkx as nx
import pandas as pd

from src.features.structured import CLAIM_KEY_COL
from src.graph.construction import CLAIM_NODE_TYPE
from src.graph.features import extract_claim_features

# Neutral fill values for a claim_key with no corresponding claim node in
# the graph (or a graph with zero claim nodes): zero centrality/sharing,
# no community membership, "not shared".
_FILL_VALUES: dict[str, object] = {
    "degree_centrality": 0.0,
    "betweenness_centrality": 0.0,
    "closeness_centrality": 0.0,
    "eigenvector_centrality": 0.0,
    "community_id": -1,
    "num_entities": 0,
    "max_entity_claim_degree": 0,
    "mean_entity_claim_degree": 0.0,
    "has_shared_entity": False,
}


def build_graph_claim_features(
    graph: nx.Graph, claim_keys: Iterable[str], claim_node_type: str = CLAIM_NODE_TYPE
) -> pd.DataFrame:
    """Build a claim-keyed graph feature table aligned to `claim_keys`.

    Args:
        graph: A `networkx.Graph` built by `src.graph.construction`
            (`build_health_graph` or `build_motor_graph`). Never mutated.
        claim_keys: The full set of `claim_key` values the caller expects a
            feature row for (e.g. `structured_df[CLAIM_KEY_COL]`). Order is
            preserved in the output. Duplicates are not deduplicated by
            this function.
        claim_node_type: The node type to treat as a claim; passed through
            to `extract_claim_features`.

    Returns:
        One row per entry of `claim_keys`, columns `claim_key` plus every
        column `extract_claim_features` produces. A `claim_key` with no
        matching claim node in `graph` (including when `graph` has zero
        nodes) gets safe neutral defaults instead of `NaN`:
        `degree_centrality`/`betweenness_centrality`/`closeness_centrality`/
        `eigenvector_centrality`/`mean_entity_claim_degree` -> `0.0`,
        `num_entities`/`max_entity_claim_degree` -> `0`, `community_id` ->
        `-1` (no community), `has_shared_entity` -> `False`.
    """
    claim_keys_list = list(claim_keys)
    template = pd.DataFrame({CLAIM_KEY_COL: claim_keys_list})

    has_claim_nodes = any(
        attrs.get("node_type") == claim_node_type for _, attrs in graph.nodes(data=True)
    )
    if not has_claim_nodes:
        # extract_claim_features raises KeyError on a graph with zero claim
        # nodes (its community-id frame ends up columnless) -- skip the
        # call entirely and go straight to defaults for every claim_key.
        merged = template.copy()
    else:
        raw = extract_claim_features(graph, claim_node_type=claim_node_type)
        raw = raw.rename(columns={"node": CLAIM_KEY_COL})
        merged = template.merge(raw, on=CLAIM_KEY_COL, how="left")

    for col, fill_value in _FILL_VALUES.items():
        if col not in merged.columns:
            merged[col] = fill_value
        merged[col] = merged[col].fillna(fill_value)

    merged["community_id"] = merged["community_id"].astype(int)
    merged["has_shared_entity"] = merged["has_shared_entity"].astype(bool)

    return merged.reset_index(drop=True)
