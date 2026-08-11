import networkx as nx
import pandas as pd

from src.data.synthetic import FraudRingConfig, assign_synthetic_entities, inject_fraud_rings
from src.features import graph as graph_features
from src.features import structured
from src.graph import construction


def _tiny_health_claims_df():
    return pd.DataFrame(
        {
            "ClaimID": ["CLM100", "CLM200", "CLM201"],
            "Provider": ["PRV1", "PRV1", "PRV2"],
            "BeneID": ["BENE1", "BENE2", "BENE3"],
            "AttendingPhysician": ["PHY1", "PHY1", "PHY2"],
        }
    )


def _augmented_motor_claims_df(n=20):
    claims = pd.DataFrame({"PolicyNumber": range(1, n + 1), "FraudFound_P": [0] * n})
    with_entities = assign_synthetic_entities(claims, seed=42, garage_pool_ratio=0.1, bank_pool_ratio=0.1)
    augmented, _ = inject_fraud_rings(
        with_entities, config=FraudRingConfig(num_rings=1, ring_size=3, seed=42)
    )
    return augmented


# ---------------------------------------------------------------------------
# Alignment against a real (non-empty) graph
# ---------------------------------------------------------------------------


def test_build_graph_claim_features_covers_all_requested_claim_keys():
    graph = construction.build_health_graph(_tiny_health_claims_df())
    claim_keys = ["claim:CLM100", "claim:CLM200", "claim:CLM201"]
    out = graph_features.build_graph_claim_features(graph, claim_keys)
    assert list(out[structured.CLAIM_KEY_COL]) == claim_keys


def test_build_graph_claim_features_preserves_requested_order():
    graph = construction.build_health_graph(_tiny_health_claims_df())
    claim_keys = ["claim:CLM201", "claim:CLM100", "claim:CLM200"]
    out = graph_features.build_graph_claim_features(graph, claim_keys)
    assert list(out[structured.CLAIM_KEY_COL]) == claim_keys


def test_build_graph_claim_features_shared_provider_has_positive_sharing_signal():
    graph = construction.build_health_graph(_tiny_health_claims_df())
    claim_keys = ["claim:CLM100", "claim:CLM200", "claim:CLM201"]
    out = graph_features.build_graph_claim_features(graph, claim_keys)
    # CLM100 and CLM200 share Provider PRV1 -> each other's claim is reachable
    shared_row = out.loc[out[structured.CLAIM_KEY_COL] == "claim:CLM100"].iloc[0]
    assert shared_row["has_shared_entity"] == True  # noqa: E712
    assert shared_row["max_entity_claim_degree"] >= 1


def test_build_graph_claim_features_no_mutation():
    graph = construction.build_health_graph(_tiny_health_claims_df())
    nodes_before = set(graph.nodes)
    edges_before = set(graph.edges)
    graph_features.build_graph_claim_features(graph, ["claim:CLM100"])
    assert set(graph.nodes) == nodes_before
    assert set(graph.edges) == edges_before


def test_build_graph_claim_features_deterministic():
    graph = construction.build_health_graph(_tiny_health_claims_df())
    claim_keys = ["claim:CLM100", "claim:CLM200", "claim:CLM201"]
    out1 = graph_features.build_graph_claim_features(graph, claim_keys)
    out2 = graph_features.build_graph_claim_features(graph, claim_keys)
    pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# Missing / absent claims handled safely
# ---------------------------------------------------------------------------


def test_build_graph_claim_features_unknown_claim_key_gets_safe_defaults():
    graph = construction.build_health_graph(_tiny_health_claims_df())
    out = graph_features.build_graph_claim_features(graph, ["claim:NOT_IN_GRAPH"])
    row = out.iloc[0]
    assert row["degree_centrality"] == 0.0
    assert row["betweenness_centrality"] == 0.0
    assert row["closeness_centrality"] == 0.0
    assert row["eigenvector_centrality"] == 0.0
    assert row["community_id"] == -1
    assert row["num_entities"] == 0
    assert row["max_entity_claim_degree"] == 0
    assert row["mean_entity_claim_degree"] == 0.0
    assert row["has_shared_entity"] == False  # noqa: E712
    assert not row.isna().any()


def test_build_graph_claim_features_empty_graph_does_not_raise():
    empty_graph = nx.Graph()
    out = graph_features.build_graph_claim_features(empty_graph, ["claim:CLM100", "claim:CLM200"])
    assert len(out) == 2
    assert not out.isna().any().any()
    assert (out["has_shared_entity"] == False).all()  # noqa: E712


def test_build_graph_claim_features_empty_claim_keys_returns_empty_frame():
    graph = construction.build_health_graph(_tiny_health_claims_df())
    out = graph_features.build_graph_claim_features(graph, [])
    assert len(out) == 0
    assert structured.CLAIM_KEY_COL in out.columns


# ---------------------------------------------------------------------------
# Motor: cross-check claim_key consistency with the structured adapter
# ---------------------------------------------------------------------------


def test_motor_graph_claim_keys_match_structured_claim_keys():
    augmented = _augmented_motor_claims_df()
    graph = construction.build_motor_graph(augmented)
    structured_out = structured.build_motor_structured_features(augmented)
    claim_keys = list(structured_out[structured.CLAIM_KEY_COL])

    out = graph_features.build_graph_claim_features(graph, claim_keys)

    assert list(out[structured.CLAIM_KEY_COL]) == claim_keys
    # every motor claim is in the graph, so nothing should fall back to defaults
    assert not out.isna().any().any()
