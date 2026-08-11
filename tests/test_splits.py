import pandas as pd
import pytest

from src.data import splits


def _tiny_stratified_df(n_per_class=20):
    rows = [{"id": i, "label": 0} for i in range(n_per_class)]
    rows += [{"id": 100 + i, "label": 1} for i in range(n_per_class)]
    return pd.DataFrame(rows)


def test_split_dataframe_is_deterministic():
    df = _tiny_stratified_df()
    train1, val1, test1 = splits.split_dataframe(df, stratify_col="label", random_state=42)
    train2, val2, test2 = splits.split_dataframe(df, stratify_col="label", random_state=42)
    assert list(train1["id"]) == list(train2["id"])
    assert list(val1["id"]) == list(val2["id"])
    assert list(test1["id"]) == list(test2["id"])


def test_split_dataframe_no_overlap_and_covers_all_rows():
    df = _tiny_stratified_df()
    train, val, test = splits.split_dataframe(df, stratify_col="label")
    train_ids, val_ids, test_ids = set(train["id"]), set(val["id"]), set(test["id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(df["id"])


def test_split_health_providers_keeps_all_claims_of_a_provider_together():
    provider_labels = pd.DataFrame(
        {
            "Provider": [f"P{i}" for i in range(40)],
            "PotentialFraud": (["Yes"] * 10 + ["No"] * 30),
        }
    )
    provider_split_map = splits.split_health_providers(
        provider_labels, provider_col="Provider", label_col="PotentialFraud"
    )
    assert set(provider_split_map.values()) <= {"train", "val", "test"}
    assert set(provider_split_map.keys()) == set(provider_labels["Provider"])

    claims = pd.DataFrame(
        {
            "ClaimID": [f"C{i}" for i in range(80)],
            "Provider": [f"P{i % 40}" for i in range(80)],
        }
    )
    claims["split"] = splits.assign_claims_to_provider_splits(
        claims, provider_col="Provider", provider_split_map=provider_split_map
    )
    # every claim from the same provider must land in the same split
    per_provider_split_counts = claims.groupby("Provider")["split"].nunique()
    assert (per_provider_split_counts == 1).all()


def test_assign_claims_to_provider_splits_raises_on_unknown_provider():
    claims = pd.DataFrame({"Provider": ["P1", "P_UNKNOWN"]})
    with pytest.raises(ValueError, match="no split assignment"):
        splits.assign_claims_to_provider_splits(
            claims, provider_col="Provider", provider_split_map={"P1": "train"}
        )
