"""Unit tests for src/features/fusion.py: final structured + graph + forensic
join into one claim-keyed, model-ready feature matrix, with the target
returned separately. Tiny in-memory fixtures only, no data/raw/ dependency."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from src.data.schemas import MOTOR_ALL_COLUMNS, MOTOR_TARGET
from src.data.synthetic import FraudRingConfig, assign_synthetic_entities, inject_fraud_rings
from src.features import fusion
from src.features.structured import CLAIM_KEY_COL
from src.forensics.ela import ELAFeatures, ELAResult
from src.graph import construction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_motor_df():
    row = {c: 0 for c in MOTOR_ALL_COLUMNS}
    row["Make"] = "Honda"
    return pd.DataFrame(
        [{**row, MOTOR_TARGET: 1}, {**row, MOTOR_TARGET: 0}, {**row, MOTOR_TARGET: 0}],
        index=[10, 11, 12],
    )


def _augmented_motor_claims_df(n=20):
    claims = pd.DataFrame({c: 0 for c in MOTOR_ALL_COLUMNS}, index=range(n))
    claims["Make"] = "Honda"
    claims[MOTOR_TARGET] = 0
    with_entities = assign_synthetic_entities(claims, seed=42, garage_pool_ratio=0.1, bank_pool_ratio=0.1)
    augmented, _ = inject_fraud_rings(
        with_entities, config=FraudRingConfig(num_rings=1, ring_size=3, seed=42)
    )
    return augmented


def _tiny_health_tables():
    inpatient = pd.DataFrame(
        {
            "ClaimID": ["CLM100"],
            "BeneID": ["BENE1"],
            "Provider": ["PRV1"],
            "AttendingPhysician": ["PHY1"],
            "ClmDiagnosisCode_1": ["D1"],
            "ClmDiagnosisCode_2": [np.nan],
        }
    )
    outpatient = pd.DataFrame(
        {
            "ClaimID": ["CLM200", "CLM201"],
            "BeneID": ["BENE1", "BENE2"],
            "Provider": ["PRV1", "PRV2"],
            "AttendingPhysician": ["PHY1", "PHY2"],
            "ClmDiagnosisCode_1": [np.nan, "D9"],
            "ClmDiagnosisCode_2": [np.nan, np.nan],
        }
    )
    beneficiary = pd.DataFrame(
        {
            "BeneID": ["BENE1", "BENE2"],
            "DOD": [None, "2009-01-01"],
            "ChronicCond_Alzheimer": [1, 2],
            "RenalDiseaseIndicator": ["0", "Y"],
        }
    )
    provider_labels = pd.DataFrame({"Provider": ["PRV1", "PRV2"], "PotentialFraud": ["Yes", "No"]})
    return inpatient, outpatient, beneficiary, provider_labels


def _make_ela_result() -> ELAResult:
    features = ELAFeatures(
        mean_error=1.5, std_error=2.5, median_error=1.0, max_error=40.0, p95_error=10.0, high_error_pixel_ratio=0.05
    )
    return ELAResult(ela_image=Image.new("RGB", (4, 4)), features=features, metadata={"original_format": "JPEG"})


# ---------------------------------------------------------------------------
# build_forensic_claim_features
# ---------------------------------------------------------------------------


def test_build_forensic_claim_features_unavailable_by_default():
    out = fusion.build_forensic_claim_features(["claim:1", "claim:2"])
    assert list(out[CLAIM_KEY_COL]) == ["claim:1", "claim:2"]
    assert (out["ela_available"] == False).all()  # noqa: E712
    assert out["ela_mean_error"].isna().all()


def test_build_forensic_claim_features_uses_bundle_when_present():
    bundle = fusion.ForensicResultBundle(ela=_make_ela_result())
    out = fusion.build_forensic_claim_features(["claim:1", "claim:2"], {"claim:1": bundle})
    row1 = out.loc[out[CLAIM_KEY_COL] == "claim:1"].iloc[0]
    row2 = out.loc[out[CLAIM_KEY_COL] == "claim:2"].iloc[0]
    assert row1["ela_available"] == True  # noqa: E712
    assert row1["ela_mean_error"] == 1.5
    assert row2["ela_available"] == False  # noqa: E712


def test_build_forensic_claim_features_empty_claim_keys():
    out = fusion.build_forensic_claim_features([])
    assert len(out) == 0
    assert CLAIM_KEY_COL in out.columns


# ---------------------------------------------------------------------------
# build_fused_features
# ---------------------------------------------------------------------------


def test_build_fused_features_motor_has_no_leakage_columns():
    claims = _augmented_motor_claims_df()
    structured_df = fusion.build_motor_structured_features(claims)
    graph = construction.build_motor_graph(claims)
    fused = fusion.build_fused_features(structured_df, graph)
    for col in (MOTOR_TARGET, "synthetic_fraud_ring_label", "synthetic_ring_id", "claim_fraud_label_provider_derived"):
        assert col not in fused.columns


def test_build_fused_features_motor_combines_all_three_sources():
    claims = _augmented_motor_claims_df()
    structured_df = fusion.build_motor_structured_features(claims)
    graph = construction.build_motor_graph(claims)
    fused = fusion.build_fused_features(structured_df, graph)
    assert "Make" in fused.columns  # structured
    assert "has_shared_entity" in fused.columns  # graph
    assert "ela_available" in fused.columns  # forensic
    assert len(fused) == len(structured_df)


def test_build_fused_features_preserves_row_count_and_order():
    claims = _augmented_motor_claims_df()
    structured_df = fusion.build_motor_structured_features(claims)
    graph = construction.build_motor_graph(claims)
    fused = fusion.build_fused_features(structured_df, graph)
    assert list(fused[CLAIM_KEY_COL]) == list(structured_df[CLAIM_KEY_COL])


def test_build_fused_features_no_source_mutation():
    claims = _augmented_motor_claims_df()
    structured_df = fusion.build_motor_structured_features(claims)
    structured_before = structured_df.copy(deep=True)
    graph = construction.build_motor_graph(claims)
    nodes_before = set(graph.nodes)
    fusion.build_fused_features(structured_df, graph)
    pd.testing.assert_frame_equal(structured_df, structured_before)
    assert set(graph.nodes) == nodes_before


def test_build_fused_features_deterministic():
    claims = _augmented_motor_claims_df()
    structured_df = fusion.build_motor_structured_features(claims)
    graph = construction.build_motor_graph(claims)
    out1 = fusion.build_fused_features(structured_df, graph)
    out2 = fusion.build_fused_features(structured_df, graph)
    pd.testing.assert_frame_equal(out1, out2)


def test_build_fused_features_rejects_missing_claim_key_column():
    df = pd.DataFrame({"a": [1, 2]})
    try:
        fusion.build_fused_features(df, construction.build_motor_graph(_augmented_motor_claims_df()))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_fused_features_rejects_duplicate_claim_keys():
    df = pd.DataFrame({CLAIM_KEY_COL: ["claim:1", "claim:1"], "x": [1, 2]})
    try:
        fusion.build_fused_features(df, construction.build_motor_graph(_augmented_motor_claims_df()))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_build_fused_features_unavailable_forensic_handled_cleanly():
    claims = _augmented_motor_claims_df()
    structured_df = fusion.build_motor_structured_features(claims)
    graph = construction.build_motor_graph(claims)
    fused = fusion.build_fused_features(structured_df, graph, forensic_results=None)
    assert (fused["ela_available"] == False).all()  # noqa: E712
    assert (fused["copy_move_available"] == False).all()  # noqa: E712
    assert (fused["metadata_available"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# Target extraction (kept separate from features)
# ---------------------------------------------------------------------------


def test_build_motor_target_claim_key_and_values():
    out = fusion.build_motor_target(_tiny_motor_df())
    assert list(out[CLAIM_KEY_COL]) == ["claim:10", "claim:11", "claim:12"]
    assert list(out[fusion.TARGET_COL]) == [1, 0, 0]


def test_build_health_target_claim_key_and_values():
    inpatient, outpatient, beneficiary, provider_labels = _tiny_health_tables()
    out = fusion.build_health_target(inpatient, outpatient, provider_labels)
    assert set(out[CLAIM_KEY_COL]) == {"claim:CLM100", "claim:CLM200", "claim:CLM201"}
    row = out.loc[out[CLAIM_KEY_COL] == "claim:CLM100"].iloc[0]
    assert row[fusion.TARGET_COL] == 1  # PRV1 -> Yes
    row2 = out.loc[out[CLAIM_KEY_COL] == "claim:CLM201"].iloc[0]
    assert row2[fusion.TARGET_COL] == 0  # PRV2 -> No


# ---------------------------------------------------------------------------
# End-to-end orchestrators: fuse_motor_claims / fuse_health_claims
# ---------------------------------------------------------------------------


def test_fuse_motor_claims_returns_features_and_target_never_mixed():
    claims = _augmented_motor_claims_df()
    graph = construction.build_motor_graph(claims)
    X, y = fusion.fuse_motor_claims(claims, graph)
    assert MOTOR_TARGET not in X.columns
    assert fusion.TARGET_COL not in X.columns
    assert len(X) == len(y) == len(claims)
    assert list(y.index) == list(X.index)


def test_fuse_motor_claims_never_includes_synthetic_ring_labels():
    claims = _augmented_motor_claims_df()
    graph = construction.build_motor_graph(claims)
    X, _ = fusion.fuse_motor_claims(claims, graph)
    assert "synthetic_fraud_ring_label" not in X.columns
    assert "synthetic_ring_id" not in X.columns


def test_fuse_motor_claims_target_matches_source():
    claims = _augmented_motor_claims_df()
    graph = construction.build_motor_graph(claims)
    X, y = fusion.fuse_motor_claims(claims, graph)
    claim_key_to_row_index = {f"claim:{idx}": idx for idx in claims.index}
    for pos in range(len(X)):
        raw_idx = claim_key_to_row_index[X.iloc[pos][CLAIM_KEY_COL]]
        assert y.iloc[pos] == claims.loc[raw_idx, MOTOR_TARGET]


def test_fuse_motor_claims_deterministic():
    claims = _augmented_motor_claims_df()
    graph = construction.build_motor_graph(claims)
    X1, y1 = fusion.fuse_motor_claims(claims, graph)
    X2, y2 = fusion.fuse_motor_claims(claims, graph)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_fuse_motor_claims_no_source_mutation():
    claims = _augmented_motor_claims_df()
    claims_before = claims.copy(deep=True)
    graph = construction.build_motor_graph(claims)
    fusion.fuse_motor_claims(claims, graph)
    pd.testing.assert_frame_equal(claims, claims_before)


def test_fuse_health_claims_returns_features_and_target_never_mixed():
    inpatient, outpatient, beneficiary, provider_labels = _tiny_health_tables()
    graph = construction.build_health_graph(
        pd.concat([inpatient, outpatient], ignore_index=True, sort=False)
    )
    X, y = fusion.fuse_health_claims(inpatient, outpatient, beneficiary, provider_labels, graph)
    assert "claim_fraud_label_provider_derived" not in X.columns
    assert fusion.TARGET_COL not in X.columns
    assert len(X) == len(y) == 3


def test_fuse_health_claims_preserves_health_specific_columns():
    inpatient, outpatient, beneficiary, provider_labels = _tiny_health_tables()
    graph = construction.build_health_graph(
        pd.concat([inpatient, outpatient], ignore_index=True, sort=False)
    )
    X, _ = fusion.fuse_health_claims(inpatient, outpatient, beneficiary, provider_labels, graph)
    assert "claim_type" in X.columns
    assert "is_deceased" in X.columns
    assert "has_shared_entity" in X.columns


def test_fuse_health_claims_target_matches_provider_derived_source():
    inpatient, outpatient, beneficiary, provider_labels = _tiny_health_tables()
    graph = construction.build_health_graph(
        pd.concat([inpatient, outpatient], ignore_index=True, sort=False)
    )
    X, y = fusion.fuse_health_claims(inpatient, outpatient, beneficiary, provider_labels, graph)
    row_pos = list(X[CLAIM_KEY_COL]).index("claim:CLM100")
    assert y.iloc[row_pos] == 1  # PRV1 -> Yes


def test_fuse_health_claims_no_source_mutation():
    inpatient, outpatient, beneficiary, provider_labels = _tiny_health_tables()
    ip_before, op_before, bene_before = inpatient.copy(deep=True), outpatient.copy(deep=True), beneficiary.copy(deep=True)
    graph = construction.build_health_graph(
        pd.concat([inpatient, outpatient], ignore_index=True, sort=False)
    )
    fusion.fuse_health_claims(inpatient, outpatient, beneficiary, provider_labels, graph)
    pd.testing.assert_frame_equal(inpatient, ip_before)
    pd.testing.assert_frame_equal(outpatient, op_before)
    pd.testing.assert_frame_equal(beneficiary, bene_before)


def test_fuse_health_claims_deterministic():
    inpatient, outpatient, beneficiary, provider_labels = _tiny_health_tables()
    graph = construction.build_health_graph(
        pd.concat([inpatient, outpatient], ignore_index=True, sort=False)
    )
    X1, y1 = fusion.fuse_health_claims(inpatient, outpatient, beneficiary, provider_labels, graph)
    X2, y2 = fusion.fuse_health_claims(inpatient, outpatient, beneficiary, provider_labels, graph)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)
