import networkx as nx
import pandas as pd
import pytest

from src.data.synthetic import (
    FraudRingConfig,
    RING_LABEL_COL,
    assign_synthetic_entities,
    inject_fraud_rings,
)
from src.graph import construction, features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_health_claims_df(n=6):
    return pd.DataFrame(
        {
            "ClaimID": [f"CLM{i:03d}" for i in range(n)],
            "Provider": ["PRV1", "PRV1", "PRV2", "PRV2", "PRV3", "PRV3"][:n],
            "BeneID": ["BENE1", "BENE2", "BENE3", "BENE1", "BENE4", "BENE5"][:n],
            "AttendingPhysician": ["PHY1", "PHY1", "PHY2", None, "PHY3", "PHY3"][:n],
            "claim_type": ["inpatient", "outpatient", "inpatient", "outpatient", "inpatient", "outpatient"][:n],
        }
    )


def _tiny_motor_claims_df(n=30):
    return pd.DataFrame(
        {
            "PolicyNumber": range(1, n + 1),
            "FraudFound_P": [0] * (n - 2) + [1, 1],
        }
    )


def _augmented_motor_claims_df(
    n=30, num_rings=2, ring_size=3, garage_pool_ratio=0.05, bank_pool_ratio=0.08
):
    claims = _tiny_motor_claims_df(n)
    with_entities = assign_synthetic_entities(
        claims, seed=42, garage_pool_ratio=garage_pool_ratio, bank_pool_ratio=bank_pool_ratio
    )
    augmented, ring_assignments = inject_fraud_rings(
        with_entities, config=FraudRingConfig(num_rings=num_rings, ring_size=ring_size, seed=42)
    )
    return augmented, ring_assignments


# ---------------------------------------------------------------------------
# Health graph construction
# ---------------------------------------------------------------------------


def test_build_health_graph_node_and_edge_counts():
    claims = _tiny_health_claims_df()
    graph = construction.build_health_graph(claims)

    claim_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == construction.CLAIM_NODE_TYPE]
    provider_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == construction.PROVIDER_NODE_TYPE]
    bene_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == construction.BENEFICIARY_NODE_TYPE]
    physician_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == construction.PHYSICIAN_NODE_TYPE]

    assert len(claim_nodes) == 6
    assert len(provider_nodes) == 3  # PRV1, PRV2, PRV3
    assert len(bene_nodes) == 5  # BENE1 shared by two claims
    assert len(physician_nodes) == 3  # PHY1, PHY2, PHY3 (one claim has no physician)

    # one claim (index 3) has no attending physician -> no physician edge
    assert graph.degree[construction._node_id("claim", "CLM003")] == 2  # provider + beneficiary only


def test_build_health_graph_shared_provider_creates_shared_neighbor():
    claims = _tiny_health_claims_df()
    graph = construction.build_health_graph(claims)
    provider_node = construction._node_id(construction.PROVIDER_NODE_TYPE, "PRV1")
    # PRV1 is shared by CLM000 and CLM001
    assert graph.degree[provider_node] == 2


def test_build_health_graph_missing_columns_raises():
    claims = _tiny_health_claims_df().drop(columns=["Provider"])
    with pytest.raises(ValueError, match="missing required columns"):
        construction.build_health_graph(claims)


def test_build_health_graph_empty_input_returns_empty_graph():
    claims = _tiny_health_claims_df(n=0)
    graph = construction.build_health_graph(claims)
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_build_health_graph_does_not_mutate_source():
    claims = _tiny_health_claims_df()
    original = claims.copy()
    construction.build_health_graph(claims)
    pd.testing.assert_frame_equal(claims, original)


def test_build_health_graph_deterministic():
    claims = _tiny_health_claims_df()
    graph_a = construction.build_health_graph(claims)
    graph_b = construction.build_health_graph(claims)
    assert sorted(graph_a.nodes) == sorted(graph_b.nodes)
    assert sorted(graph_a.edges) == sorted(graph_b.edges)


# ---------------------------------------------------------------------------
# Motor graph construction
# ---------------------------------------------------------------------------


def test_build_motor_graph_node_and_edge_counts():
    augmented, _ = _augmented_motor_claims_df(n=30)
    graph = construction.build_motor_graph(augmented)

    claim_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == construction.CLAIM_NODE_TYPE]
    garage_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == construction.GARAGE_NODE_TYPE]
    bank_nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == construction.BANK_ACCOUNT_NODE_TYPE]

    assert len(claim_nodes) == 30
    assert graph.number_of_edges() == 30 * 2  # each claim -> 1 garage edge + 1 bank edge
    assert len(garage_nodes) >= 1
    assert len(bank_nodes) >= 1


def test_build_motor_graph_missing_columns_raises():
    claims = _tiny_motor_claims_df()
    with pytest.raises(ValueError, match="missing required columns"):
        construction.build_motor_graph(claims)


def test_build_motor_graph_empty_input_returns_empty_graph():
    claims = assign_synthetic_entities(_tiny_motor_claims_df(n=5), seed=42).iloc[0:0]
    graph = construction.build_motor_graph(claims)
    assert graph.number_of_nodes() == 0


def test_build_motor_graph_does_not_mutate_source():
    augmented, _ = _augmented_motor_claims_df(n=30)
    original = augmented.copy()
    construction.build_motor_graph(augmented)
    pd.testing.assert_frame_equal(augmented, original)


def test_build_motor_graph_carries_ring_labels_as_node_attrs():
    augmented, _ = _augmented_motor_claims_df(n=30, num_rings=2, ring_size=3)
    graph = construction.build_motor_graph(augmented)
    ring_member_claim = augmented[augmented[RING_LABEL_COL] == 1].index[0]
    claim_node = construction._node_id(construction.CLAIM_NODE_TYPE, ring_member_claim)
    assert graph.nodes[claim_node][RING_LABEL_COL] == 1


# ---------------------------------------------------------------------------
# Centrality features
# ---------------------------------------------------------------------------


def test_compute_centrality_features_returns_row_per_node():
    claims = _tiny_health_claims_df()
    graph = construction.build_health_graph(claims)
    result = features.compute_centrality_features(graph)
    assert len(result) == graph.number_of_nodes()
    assert set(result["node"]) == set(graph.nodes)
    assert {"degree_centrality", "betweenness_centrality", "closeness_centrality", "eigenvector_centrality"} <= set(
        result.columns
    )


def test_compute_centrality_features_empty_graph():
    result = features.compute_centrality_features(nx.Graph())
    assert result.empty
    assert "degree_centrality" in result.columns


def test_compute_centrality_features_deterministic():
    claims = _tiny_health_claims_df()
    graph = construction.build_health_graph(claims)
    a = features.compute_centrality_features(graph).sort_values("node").reset_index(drop=True)
    b = features.compute_centrality_features(graph).sort_values("node").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_shared_provider_has_higher_degree_centrality_than_isolated_claim_neighbor():
    # PRV1 is shared by two claims; PRV3 is shared by two claims too, but a
    # beneficiary appearing once has fewer connections than a hub provider
    # once multiple claims funnel through it.
    claims = _tiny_health_claims_df()
    graph = construction.build_health_graph(claims)
    result = features.compute_centrality_features(graph).set_index("node")
    provider_node = construction._node_id(construction.PROVIDER_NODE_TYPE, "PRV1")
    single_bene_node = construction._node_id(construction.BENEFICIARY_NODE_TYPE, "BENE3")
    assert result.loc[provider_node, "degree_centrality"] > result.loc[single_bene_node, "degree_centrality"]


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------


def test_detect_communities_returns_mapping_for_every_node():
    claims = _tiny_health_claims_df()
    graph = construction.build_health_graph(claims)
    communities = features.detect_communities(graph)
    assert set(communities.keys()) == set(graph.nodes)


def test_detect_communities_empty_graph():
    assert features.detect_communities(nx.Graph()) == {}


def test_detect_communities_deterministic():
    claims = _tiny_health_claims_df()
    graph = construction.build_health_graph(claims)
    a = features.detect_communities(graph)
    b = features.detect_communities(graph)
    assert a == b


def test_detect_communities_separates_disconnected_components():
    # two fully disconnected claim clusters should land in different communities
    claims = pd.DataFrame(
        {
            "ClaimID": ["A1", "A2", "B1", "B2"],
            "Provider": ["PRV_A", "PRV_A", "PRV_B", "PRV_B"],
            "BeneID": ["BENE_A1", "BENE_A2", "BENE_B1", "BENE_B2"],
            "AttendingPhysician": [None, None, None, None],
        }
    )
    graph = construction.build_health_graph(claims)
    communities = features.detect_communities(graph)
    claim_a1 = construction._node_id("claim", "A1")
    claim_b1 = construction._node_id("claim", "B1")
    assert communities[claim_a1] != communities[claim_b1]


# ---------------------------------------------------------------------------
# Entity-sharing / fraud-ring indicator features
# ---------------------------------------------------------------------------


def test_compute_entity_sharing_features_flags_shared_entities():
    claims = _tiny_health_claims_df()
    graph = construction.build_health_graph(claims)
    result = features.compute_entity_sharing_features(graph).set_index("node")

    claim_with_shared_provider = construction._node_id("claim", "CLM000")  # shares PRV1
    claim_fully_isolated_entities = construction._node_id("claim", "CLM004")  # PRV3 shared, but check CLM002

    assert result.loc[claim_with_shared_provider, "has_shared_entity"]
    assert result.loc[claim_with_shared_provider, "max_entity_claim_degree"] >= 1


def test_compute_entity_sharing_features_empty_graph():
    result = features.compute_entity_sharing_features(nx.Graph())
    assert result.empty


def test_extract_claim_features_only_returns_claim_rows():
    claims = _tiny_health_claims_df()
    graph = construction.build_health_graph(claims)
    result = features.extract_claim_features(graph)
    assert len(result) == 6
    expected_nodes = {construction._node_id("claim", cid) for cid in claims["ClaimID"]}
    assert set(result["node"]) == expected_nodes
    assert {"degree_centrality", "community_id", "max_entity_claim_degree"} <= set(result.columns)


# ---------------------------------------------------------------------------
# Synthetic fraud-ring recovery validation
# ---------------------------------------------------------------------------


def test_injected_ring_members_recover_higher_shared_entity_degree_than_non_members():
    # Sparse baseline entity pools (most claims get a near-unique garage/bank)
    # so the deliberately shared ring entities stand out against the
    # background collision rate, isolating the ring signal from baseline
    # pool-size noise.
    augmented, ring_assignments = _augmented_motor_claims_df(
        n=60, num_rings=3, ring_size=4, garage_pool_ratio=0.9, bank_pool_ratio=0.9
    )
    graph = construction.build_motor_graph(augmented)
    claim_features = features.extract_claim_features(graph)

    ring_member_indices = augmented[augmented[RING_LABEL_COL] == 1].index
    ring_member_nodes = {construction._node_id("claim", idx) for idx in ring_member_indices}

    ring_rows = claim_features[claim_features["node"].isin(ring_member_nodes)]
    non_ring_rows = claim_features[~claim_features["node"].isin(ring_member_nodes)]

    assert ring_rows["max_entity_claim_degree"].mean() > non_ring_rows["max_entity_claim_degree"].mean()


def test_injected_ring_members_cluster_into_shared_communities():
    augmented, ring_assignments = _augmented_motor_claims_df(n=60, num_rings=3, ring_size=4)
    graph = construction.build_motor_graph(augmented)
    communities = features.detect_communities(graph)

    for ring_id, group in ring_assignments.groupby("ring_id"):
        claim_nodes = [construction._node_id("claim", row.claim_row_index) for row in group.itertuples()]
        community_ids = {communities[node] for node in claim_nodes}
        # all members sharing a garage (base link) must land in one community
        assert len(community_ids) == 1
