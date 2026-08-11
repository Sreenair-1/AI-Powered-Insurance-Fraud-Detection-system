"""Graph-derived features: centrality, community detection, and fraud-ring
indicators, plus a reusable claim-level feature extraction API.

Every function here reads a `networkx.Graph` built by `src.graph.construction`
— nothing here decides fraud. `compute_centrality_features` additionally may
*attach* a small results cache to the graph's own `graph.graph` metadata dict
(see below) so it is no longer a strictly side-effect-free read when running
in approximate mode; it never touches node/edge structure.
"""
from __future__ import annotations

import random
from typing import Hashable, Optional

import networkx as nx
import pandas as pd

from src.graph.construction import CLAIM_NODE_TYPE, NODE_TYPE_ATTR

COMMUNITY_ID_ATTR = "community_id"

# Exact betweenness/closeness centrality are O(V*(V+E)) -- measured to take
# ~46s at 2,000 motor claims and to extrapolate to 45min+ at the full 15,420
# motor / ~558k health claim scale (see docs/app_ui.md, docs/graph.md).
# Above this many nodes, compute_centrality_features() switches to the
# approximate estimators below by default.
LARGE_GRAPH_NODE_THRESHOLD = 1500

# k-sampled sources for approximate betweenness (networkx's own `k=` param)
# and landmark count for approximate closeness (see
# `_approximate_closeness_centrality`). Larger values trade speed for
# accuracy; these defaults keep large-graph runs in the seconds-to-low-tens-
# of-seconds range instead of tens of minutes.
DEFAULT_BETWEENNESS_SAMPLE_SIZE = 200
DEFAULT_CLOSENESS_LANDMARKS = 200
DEFAULT_APPROXIMATION_SEED = 42
# Approximate-mode eigenvector centrality still uses power iteration (the
# numpy/ARPACK variant requires a single connected component, which these
# entity graphs never are), but fails fast to the existing NaN fallback
# instead of spending up to 1000 iterations on a graph too large to converge
# quickly.
APPROXIMATE_EIGENVECTOR_MAX_ITER = 100

_CENTRALITY_CACHE_ATTR = "_centrality_feature_cache"


def _approximate_closeness_centrality(
    graph: nx.Graph, k: int, seed: int
) -> dict[Hashable, float]:
    """Landmark-sampled closeness centrality estimate for large graphs.

    Exact closeness centrality (`nx.closeness_centrality`) runs a
    single-source shortest-path search from *every* node: O(V*(V+E)). This
    instead runs it from only `k` random landmark nodes and estimates each
    node's closeness from its mean distance to those landmarks --
    `closeness(v) ~= 1 / mean(distance(v, landmark))`, matching the exact
    definition `closeness(v) = (n-1) / sum(distances)` when landmarks are
    representative of the graph. Cost: O(k*(V+E)).

    This is a ranking estimate, not the exact Wasserman-Faust-adjusted value
    `nx.closeness_centrality` returns for disconnected graphs -- acceptable
    here because these features feed an XGBoost model as evidence, never a
    thresholded fraud decision. A node with no reachable
    landmark (an isolated component none of the `k` samples landed in) gets
    `0.0`, the same convention `nx.closeness_centrality` uses for isolated
    nodes.
    """
    nodes = list(graph.nodes)
    if not nodes:
        return {}

    rng = random.Random(seed)
    landmarks = rng.sample(nodes, min(k, len(nodes)))
    landmark_distances = {
        landmark: nx.single_source_shortest_path_length(graph, landmark) for landmark in landmarks
    }

    closeness: dict[Hashable, float] = {}
    for node in nodes:
        distances = [
            landmark_distances[landmark][node]
            for landmark in landmarks
            if landmark != node and node in landmark_distances[landmark]
        ]
        if not distances:
            closeness[node] = 0.0
            continue
        mean_distance = sum(distances) / len(distances)
        closeness[node] = 0.0 if mean_distance == 0 else 1.0 / mean_distance
    return closeness


def compute_centrality_features(
    graph: nx.Graph,
    *,
    approximate: Optional[bool] = None,
    large_graph_threshold: int = LARGE_GRAPH_NODE_THRESHOLD,
    betweenness_sample_size: int = DEFAULT_BETWEENNESS_SAMPLE_SIZE,
    closeness_landmarks: int = DEFAULT_CLOSENESS_LANDMARKS,
    seed: int = DEFAULT_APPROXIMATION_SEED,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Compute degree/betweenness/closeness/eigenvector centrality per node.

    Returns one row per node with columns `node`, `node_type`, and the four
    centrality measures. Eigenvector centrality is set to `NaN` for a node
    when power iteration fails to converge (common on small/disconnected
    graphs) rather than raising.

    For graphs with more than `large_graph_threshold` nodes (default
    `LARGE_GRAPH_NODE_THRESHOLD`), betweenness and closeness switch
    automatically to k-sampled/landmark-based approximations (see
    `_approximate_closeness_centrality`) so the full dataset finishes in
    interactive time instead of the tens of minutes exact computation would
    take (measured; see `docs/app_ui.md`). Pass `approximate=True`/`False`
    to force the mode regardless of graph size. Below the threshold, this
    function is byte-for-byte the same exact computation as before --
    existing callers on small/medium graphs (all current tests) see no
    behavior change.

    Approximate-mode results are cached on the `graph` object itself
    (`graph.graph["_centrality_feature_cache"]`), keyed by the approximation
    parameters, so repeat calls on the same graph instance (e.g. a
    `st.cache_data`-cached Streamlit dataset rebuilt across reruns) skip
    recomputation. Exact-mode calls are never cached (unnecessary -- they
    are already fast enough on the small/medium graphs that take that path).
    Pass `use_cache=False` to bypass. The cache is deterministic for a given
    graph + parameter set, so `compute_centrality_features(graph)` called
    twice in a row still returns identical results either way.
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

    use_approximate = (
        graph.number_of_nodes() > large_graph_threshold if approximate is None else approximate
    )

    cache_key = (betweenness_sample_size, closeness_landmarks, seed)
    if use_approximate and use_cache:
        cached = graph.graph.get(_CENTRALITY_CACHE_ATTR, {}).get(cache_key)
        if cached is not None:
            return cached.copy()

    degree = nx.degree_centrality(graph)
    if use_approximate:
        k_betweenness = min(betweenness_sample_size, graph.number_of_nodes())
        betweenness = nx.betweenness_centrality(graph, k=k_betweenness, seed=seed, normalized=True)
        closeness = _approximate_closeness_centrality(
            graph, k=min(closeness_landmarks, graph.number_of_nodes()), seed=seed
        )
        eigenvector_max_iter = APPROXIMATE_EIGENVECTOR_MAX_ITER
    else:
        betweenness = nx.betweenness_centrality(graph)
        closeness = nx.closeness_centrality(graph)
        eigenvector_max_iter = 1000

    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=eigenvector_max_iter)
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
    result = pd.DataFrame(rows, columns=columns)

    if use_approximate and use_cache:
        graph.graph.setdefault(_CENTRALITY_CACHE_ATTR, {})[cache_key] = result.copy()

    return result


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


def extract_claim_features(
    graph: nx.Graph, claim_node_type: str = CLAIM_NODE_TYPE, **centrality_kwargs
) -> pd.DataFrame:
    """Reusable claim-level graph feature table for downstream feature fusion.

    Joins centrality, community id, and entity-sharing features for every
    claim node into a single DataFrame keyed by `node`. This is the API the
    feature-fusion stage calls rather than re-deriving graph features itself.

    `**centrality_kwargs` (e.g. `approximate=True`, `betweenness_sample_size=`)
    are forwarded to `compute_centrality_features` unchanged; the default
    (nothing passed) preserves that function's own default of auto-switching
    to approximate centrality only above `LARGE_GRAPH_NODE_THRESHOLD` nodes.
    """
    centrality = compute_centrality_features(graph, **centrality_kwargs)
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
