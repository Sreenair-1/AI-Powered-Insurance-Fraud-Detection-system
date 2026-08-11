"""Graph-derived features: centrality, community detection, and fraud-ring
indicators, plus a reusable claim-level feature extraction API.

Every function here is a pure read of a `networkx.Graph` built by
`src.graph.construction` — nothing here decides fraud.
"""
from __future__ import annotations

from typing import Hashable

import networkx as nx
import pandas as pd

from src.graph.construction import CLAIM_NODE_TYPE, NODE_TYPE_ATTR

COMMUNITY_ID_ATTR = "community_id"


def compute_centrality_features(graph: nx.Graph) -> pd.DataFrame:
    """Compute degree/betweenness/closeness/eigenvector centrality per node.

    Returns one row per node with columns `node`, `node_type`, and the four
    centrality measures. Deterministic for a given graph. Eigenvector
    centrality is set to `NaN` for a node when power iteration fails to
    converge (common on small/disconnected graphs) rather than raising.
    """
    columns = [
        "node",
        "node_type",
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
    ]
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=columns)

    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
    except (nx.PowerIterationFailedConvergence, nx.AmbiguousSolution, ZeroDivisionError):
        eigenvector = {node: float("nan") for node in graph.nodes}

    rows = [
        {
            "node": node,
            "node_type": graph.nodes[node].get(NODE_TYPE_ATTR),
            "degree_centrality": degree[node],
            "betweenness_centrality": betweenness[node],
            "closeness_centrality": closeness[node],
            "eigenvector_centrality": eigenvector.get(node, float("nan")),
        }
        for node in graph.nodes
    ]
    return pd.DataFrame(rows, columns=columns)


def detect_communities(graph: nx.Graph) -> dict[Hashable, int]:
    """Assign each node an integer community id via greedy modularity maximization.

    Deterministic for a given graph (no random seed involved). Isolated
    nodes and singleton components each land in their own community.
    Returns `{}` for an empty graph.
    """
    if graph.number_of_nodes() == 0:
        return {}

    communities = nx.algorithms.community.greedy_modularity_communities(graph)
    return {
        node: community_idx
        for community_idx, community_nodes in enumerate(communities)
        for node in community_nodes
    }


def compute_entity_sharing_features(
    graph: nx.Graph, claim_node_type: str = CLAIM_NODE_TYPE
) -> pd.DataFrame:
    """Compute, per claim node, how much its linked entities are reused by other claims.

    This is the graph-analytics fraud-ring signal that claims
    whose garage/bank-account/provider/physician/beneficiary is also linked
    to many *other* claims are more suspicious, without asserting fraud.
    Returns one row per claim node with:

    - `num_entities`: number of distinct non-claim neighbors.
    - `max_entity_claim_degree` / `mean_entity_claim_degree`: the largest /
      average number of *other* claim nodes reachable through any one
      shared entity neighbor (0 if no entity is shared with another claim).
    - `has_shared_entity`: convenience boolean, `max_entity_claim_degree > 0`.
    """
    columns = [
        "node",
        "num_entities",
        "max_entity_claim_degree",
        "mean_entity_claim_degree",
        "has_shared_entity",
    ]
    claim_nodes = [
        node for node, attrs in graph.nodes(data=True) if attrs.get(NODE_TYPE_ATTR) == claim_node_type
    ]
    if not claim_nodes:
        return pd.DataFrame(columns=columns)

    rows = []
    for claim_node in claim_nodes:
        entity_neighbors = list(graph.neighbors(claim_node))
        other_claim_counts = []
        for entity_node in entity_neighbors:
            other_claims = [
                n
                for n in graph.neighbors(entity_node)
                if n != claim_node and graph.nodes[n].get(NODE_TYPE_ATTR) == claim_node_type
            ]
            other_claim_counts.append(len(other_claims))
        max_degree = max(other_claim_counts) if other_claim_counts else 0
        mean_degree = (sum(other_claim_counts) / len(other_claim_counts)) if other_claim_counts else 0.0
        rows.append(
            {
                "node": claim_node,
                "num_entities": len(entity_neighbors),
                "max_entity_claim_degree": max_degree,
                "mean_entity_claim_degree": mean_degree,
                "has_shared_entity": max_degree > 0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def extract_claim_features(graph: nx.Graph, claim_node_type: str = CLAIM_NODE_TYPE) -> pd.DataFrame:
    """Reusable claim-level graph feature table for downstream feature fusion.

    Joins centrality, community id, and entity-sharing features for every
    claim node into a single DataFrame keyed by `node`. This is the API the
    (future) feature-fusion stage should call rather than re-deriving graph
    features itself.
    """
    centrality = compute_centrality_features(graph)
    claim_centrality = centrality[centrality["node_type"] == claim_node_type].drop(
        columns=["node_type"]
    )

    communities = detect_communities(graph)
    community_df = pd.DataFrame(
        [{"node": node, COMMUNITY_ID_ATTR: community_id} for node, community_id in communities.items()]
    )

    sharing = compute_entity_sharing_features(graph, claim_node_type=claim_node_type)

    result = claim_centrality.merge(community_df, on="node", how="left")
    result = result.merge(sharing, on="node", how="left")
    return result.reset_index(drop=True)
