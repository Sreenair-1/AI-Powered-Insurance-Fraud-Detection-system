import pandas as pd
import pytest

from src.data import synthetic
from src.data.schemas import MOTOR_TARGET


def _tiny_claims_df(n=40):
    return pd.DataFrame(
        {
            "PolicyNumber": range(1, n + 1),
            MOTOR_TARGET: [0] * (n - 2) + [1, 1],
        }
    )


def _tiny_demographics_df(n=10):
    return pd.DataFrame(
        {
            "age": [25 + i for i in range(n)],
            "weight": [60 + i for i in range(n)],
            "height": [160 + i for i in range(n)],
            "income_lpa": [5.0 + i for i in range(n)],
            "smoker": ["no"] * n,
            "city": ["Mumbai"] * n,
            "occupation": ["engineer"] * n,
            "insurance_premium_category": ["medium"] * n,
        }
    )


# ---------------------------------------------------------------------------
# Demographic sampling
# ---------------------------------------------------------------------------


def test_sample_indian_demographics_is_deterministic():
    demo = _tiny_demographics_df()
    a = synthetic.sample_indian_demographics(20, seed=42, demographics_df=demo)
    b = synthetic.sample_indian_demographics(20, seed=42, demographics_df=demo)
    pd.testing.assert_frame_equal(a, b)


def test_sample_indian_demographics_different_seed_can_differ():
    demo = _tiny_demographics_df()
    a = synthetic.sample_indian_demographics(20, seed=1, demographics_df=demo)
    b = synthetic.sample_indian_demographics(20, seed=2, demographics_df=demo)
    assert not a["synthetic_demographic_age"].equals(b["synthetic_demographic_age"])


def test_sample_indian_demographics_prefixes_columns_and_never_mutates_input():
    demo = _tiny_demographics_df()
    original = demo.copy()
    sample = synthetic.sample_indian_demographics(5, seed=42, demographics_df=demo)
    assert all(c.startswith("synthetic_demographic_") or c == "synthetic_profile_id" for c in sample.columns)
    pd.testing.assert_frame_equal(demo, original)


def test_sample_indian_demographics_rejects_non_positive_n():
    demo = _tiny_demographics_df()
    with pytest.raises(ValueError):
        synthetic.sample_indian_demographics(0, demographics_df=demo)
    with pytest.raises(ValueError):
        synthetic.sample_indian_demographics(-3, demographics_df=demo)


# ---------------------------------------------------------------------------
# Entity ID generation
# ---------------------------------------------------------------------------


def test_generate_garage_ids_unique_and_non_empty():
    ids = synthetic.generate_garage_ids(15)
    assert len(ids) == 15
    assert len(set(ids)) == 15
    assert all(isinstance(i, str) and i for i in ids)


def test_generate_bank_account_ids_unique_and_non_empty():
    ids = synthetic.generate_bank_account_ids(15)
    assert len(ids) == 15
    assert len(set(ids)) == 15
    assert all(isinstance(i, str) and i for i in ids)


def test_generate_ids_rejects_non_positive_n():
    with pytest.raises(ValueError):
        synthetic.generate_garage_ids(0)
    with pytest.raises(ValueError):
        synthetic.generate_bank_account_ids(-1)


# ---------------------------------------------------------------------------
# Baseline entity assignment
# ---------------------------------------------------------------------------


def test_assign_synthetic_entities_deterministic_and_input_unchanged():
    claims = _tiny_claims_df()
    original = claims.copy()
    a = synthetic.assign_synthetic_entities(claims, seed=42)
    b = synthetic.assign_synthetic_entities(claims, seed=42)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(claims, original)


def test_assign_synthetic_entities_adds_expected_columns():
    claims = _tiny_claims_df()
    out = synthetic.assign_synthetic_entities(claims, seed=42)
    assert synthetic.GARAGE_ID_COL in out.columns
    assert synthetic.BANK_ACCOUNT_ID_COL in out.columns
    assert out[synthetic.GARAGE_ID_COL].notna().all()
    assert out[synthetic.BANK_ACCOUNT_ID_COL].notna().all()


def test_assign_synthetic_entities_rejects_empty_df():
    with pytest.raises(ValueError):
        synthetic.assign_synthetic_entities(pd.DataFrame())


# ---------------------------------------------------------------------------
# Fraud-ring injection
# ---------------------------------------------------------------------------


def test_fraud_ring_config_rejects_invalid_params():
    with pytest.raises(ValueError):
        synthetic.FraudRingConfig(num_rings=0)
    with pytest.raises(ValueError):
        synthetic.FraudRingConfig(ring_size=1)
    with pytest.raises(ValueError):
        synthetic.FraudRingConfig(linkage_strength=1.5)
    with pytest.raises(ValueError):
        synthetic.FraudRingConfig(linkage_strength=-0.1)
    with pytest.raises(ValueError):
        synthetic.FraudRingConfig(max_affected_proportion=0.0)


def test_inject_fraud_rings_requires_entity_columns_first():
    claims = _tiny_claims_df()
    with pytest.raises(ValueError, match="assign_synthetic_entities"):
        synthetic.inject_fraud_rings(claims, config=synthetic.FraudRingConfig(num_rings=1, ring_size=2))


def test_inject_fraud_rings_raises_when_rings_exceed_available_claims():
    claims = synthetic.assign_synthetic_entities(_tiny_claims_df(n=10), seed=42)
    config = synthetic.FraudRingConfig(num_rings=5, ring_size=4, max_affected_proportion=1.0)
    with pytest.raises(ValueError):
        synthetic.inject_fraud_rings(claims, config=config)


def test_inject_fraud_rings_raises_when_exceeding_max_affected_proportion():
    claims = synthetic.assign_synthetic_entities(_tiny_claims_df(n=100), seed=42)
    config = synthetic.FraudRingConfig(
        num_rings=5, ring_size=4, max_affected_proportion=0.1  # 20/100 = 0.2 > 0.1
    )
    with pytest.raises(ValueError, match="max_affected_proportion"):
        synthetic.inject_fraud_rings(claims, config=config)


def test_inject_fraud_rings_deterministic():
    claims = synthetic.assign_synthetic_entities(_tiny_claims_df(n=40), seed=42)
    config = synthetic.FraudRingConfig(num_rings=2, ring_size=3, seed=42)
    out1, assign1 = synthetic.inject_fraud_rings(claims, config=config)
    out2, assign2 = synthetic.inject_fraud_rings(claims, config=config)
    pd.testing.assert_frame_equal(out1, out2)
    pd.testing.assert_frame_equal(assign1, assign2)


def test_inject_fraud_rings_does_not_mutate_input():
    claims = synthetic.assign_synthetic_entities(_tiny_claims_df(n=40), seed=42)
    original = claims.copy()
    synthetic.inject_fraud_rings(claims, config=synthetic.FraudRingConfig(num_rings=2, ring_size=3))
    pd.testing.assert_frame_equal(claims, original)


def test_inject_fraud_rings_correct_ring_sizes_and_shared_garage():
    claims = synthetic.assign_synthetic_entities(_tiny_claims_df(n=50), seed=42)
    config = synthetic.FraudRingConfig(num_rings=3, ring_size=4, linkage_strength=1.0, seed=42)
    out, ring_assignments = synthetic.inject_fraud_rings(claims, config=config)

    assert len(ring_assignments) == config.num_rings * config.ring_size
    assert out[synthetic.RING_LABEL_COL].sum() == config.num_rings * config.ring_size

    for ring_id, group in ring_assignments.groupby("ring_id"):
        assert len(group) == config.ring_size
        assert group["garage_id"].nunique() == 1


def test_inject_fraud_rings_full_linkage_shares_bank_account_too():
    claims = synthetic.assign_synthetic_entities(_tiny_claims_df(n=50), seed=42)
    config = synthetic.FraudRingConfig(num_rings=2, ring_size=4, linkage_strength=1.0, seed=42)
    out, ring_assignments = synthetic.inject_fraud_rings(claims, config=config)

    for ring_id, group in ring_assignments.groupby("ring_id"):
        assert group["bank_account_id"].nunique() == 1
        assert group["shares_bank_account"].all()


def test_inject_fraud_rings_zero_linkage_never_shares_bank_account():
    claims = synthetic.assign_synthetic_entities(_tiny_claims_df(n=50), seed=42)
    config = synthetic.FraudRingConfig(num_rings=2, ring_size=4, linkage_strength=0.0, seed=42)
    _, ring_assignments = synthetic.inject_fraud_rings(claims, config=config)
    assert not ring_assignments["shares_bank_account"].any()


def test_inject_fraud_rings_does_not_touch_real_fraud_label():
    claims = synthetic.assign_synthetic_entities(_tiny_claims_df(n=40), seed=42)
    original_target = claims[MOTOR_TARGET].copy()
    out, _ = synthetic.inject_fraud_rings(claims, config=synthetic.FraudRingConfig(num_rings=2, ring_size=3))
    pd.testing.assert_series_equal(out[MOTOR_TARGET], original_target)


def test_inject_fraud_rings_labels_only_a_small_proportion():
    claims = synthetic.assign_synthetic_entities(_tiny_claims_df(n=200), seed=42)
    config = synthetic.FraudRingConfig(num_rings=3, ring_size=4, seed=42)
    out, _ = synthetic.inject_fraud_rings(claims, config=config)
    assert out[synthetic.RING_LABEL_COL].sum() / len(out) < 0.1


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_build_motor_synthetic_augmentation_end_to_end():
    claims = _tiny_claims_df(n=60)
    demo = _tiny_demographics_df()
    config = synthetic.FraudRingConfig(num_rings=2, ring_size=3, seed=42)

    result = synthetic.build_motor_synthetic_augmentation(
        motor_df=claims, ring_config=config, seed=42, demographics_df=demo
    )

    assert synthetic.GARAGE_ID_COL in result.claims.columns
    assert synthetic.BANK_ACCOUNT_ID_COL in result.claims.columns
    assert synthetic.RING_LABEL_COL in result.claims.columns
    assert len(result.ring_assignments) == config.num_rings * config.ring_size
    assert len(result.ring_demographic_profiles) == config.num_rings
    assert "ring_id" in result.ring_demographic_profiles.columns
    # demographics are never merged directly into the claims table
    assert not any(c.startswith("synthetic_demographic_") for c in result.claims.columns)


def test_build_motor_synthetic_augmentation_is_deterministic():
    claims = _tiny_claims_df(n=60)
    demo = _tiny_demographics_df()
    config = synthetic.FraudRingConfig(num_rings=2, ring_size=3, seed=42)

    r1 = synthetic.build_motor_synthetic_augmentation(
        motor_df=claims, ring_config=config, seed=42, demographics_df=demo
    )
    r2 = synthetic.build_motor_synthetic_augmentation(
        motor_df=claims, ring_config=config, seed=42, demographics_df=demo
    )
    pd.testing.assert_frame_equal(r1.claims, r2.claims)
    pd.testing.assert_frame_equal(r1.ring_assignments, r2.ring_assignments)
    pd.testing.assert_frame_equal(r1.ring_demographic_profiles, r2.ring_demographic_profiles)


def test_save_synthetic_augmentation_writes_expected_files(tmp_path):
    claims = _tiny_claims_df(n=40)
    demo = _tiny_demographics_df()
    config = synthetic.FraudRingConfig(num_rings=1, ring_size=2, seed=42)
    result = synthetic.build_motor_synthetic_augmentation(
        motor_df=claims, ring_config=config, seed=42, demographics_df=demo
    )

    paths = synthetic.save_synthetic_augmentation(result, output_dir=tmp_path / "synthetic_out")
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
