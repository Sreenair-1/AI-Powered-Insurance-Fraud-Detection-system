"""Train fraud-risk models with train/validation/test separation.

Validation drives model/threshold choices; the test partition is scored once
after choices are frozen. Motor training compares split-local synthetic graph
features to structured features, then excludes the graph if validation does
not support it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.loaders import DatasetFileMissingError, load_health_train_tables, load_motor_claims
from src.data.schemas import (HEALTH_PROVIDER_COL, HEALTH_PROVIDER_LABEL_COL, MOTOR_TARGET, PROJECT_ROOT)
from src.data.splits import assign_claims_to_provider_splits, split_dataframe, split_health_providers
from src.features.fusion import build_health_target
from src.features.structured import build_health_structured_features, build_motor_structured_features
from src.features.graph import build_graph_claim_features
from src.data.synthetic import assign_synthetic_entities
from src.graph.construction import build_motor_graph
from src.models.evaluation import compute_classification_metrics, compute_confusion_matrix, curve_data, select_threshold
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.models.persistence import save_health_model, save_motor_model

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"


def _write_json(directory: Path, name: str, payload) -> None:
    (directory / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fit_validation_candidate(model_factory, candidates, X_train, y_train, X_val, y_val):
    """Small, fixed candidate sweep ranked by validation average precision."""
    results = []
    for params in candidates:
        model = model_factory(params).fit(X_train, y_train)
        metrics = compute_classification_metrics(y_val, model.predict_proba(X_val), 0.5)
        results.append((metrics.average_precision, params, model))
    score, params, model = max(results, key=lambda row: row[0])
    return model, {"selection_metric": "validation_average_precision", "candidates": [
        {"parameters": p, "validation_average_precision": s} for s, p, _ in results
    ], "selected_parameters": params, "selected_validation_average_precision": score}


def _motor_features_for_split(claims: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Structured features plus graph features computed inside this split only."""
    entities = assign_synthetic_entities(claims, seed=seed)
    graph = build_motor_graph(entities)
    structured = build_motor_structured_features(entities)
    graph_features = build_graph_claim_features(graph, structured["claim_key"])
    return structured.merge(graph_features, on="claim_key", validate="one_to_one")


def _evaluate_and_save(model, X_val, y_val, X_test, y_test, directory: Path, objective: str,
                       min_precision: float | None, min_recall: float | None,
                       review_capacity: int | None) -> None:
    p_val = model.predict_proba(X_val)
    selection = select_threshold(y_val, p_val, objective, min_precision=min_precision,
                                 min_recall=min_recall, review_capacity=review_capacity)
    val_metrics = compute_classification_metrics(y_val, p_val, selection.threshold)
    val_metrics_baseline = compute_classification_metrics(y_val, p_val, 0.5)
    p_test = model.predict_proba(X_test)
    test_metrics = compute_classification_metrics(y_test, p_test, selection.threshold)
    test_baseline = compute_classification_metrics(y_test, p_test, 0.5)
    roc_data, pr_data = curve_data(y_val, p_val)
    _write_json(directory, "eval_metrics.json", val_metrics.to_dict())
    _write_json(directory, "validation_metrics.json", {
        "split": "validation", "threshold_selection": selection.to_dict(),
        "selected_threshold_metrics": val_metrics.to_dict(),
        "baseline_0_50_metrics": val_metrics_baseline.to_dict(),
    })
    _write_json(directory, "threshold_selection.json", selection.to_dict())
    _write_json(directory, "test_metrics.json", {
        "split": "untouched_test", "threshold_frozen_from_validation": selection.threshold,
        "selected_threshold_metrics": test_metrics.to_dict(),
        "baseline_0_50_metrics": test_baseline.to_dict(),
        "confusion_matrix": compute_confusion_matrix(y_test, p_test, selection.threshold).to_dict(),
    })
    _write_json(directory, "confusion_matrix.json",
                compute_confusion_matrix(y_val, p_val, selection.threshold).to_dict())
    _write_json(directory, "roc_curve.json", roc_data)
    _write_json(directory, "pr_curve.json", pr_data)
    print(f"validation threshold selection: {selection.to_dict()}")
    print(f"validation baseline @0.50: {val_metrics_baseline.to_dict()}")
    print(f"untouched test @frozen threshold: {test_metrics.to_dict()}")


def train_motor(output_dir: Path, sample_size: int | None, seed: int, objective: str = "max_f1",
                min_precision: float | None = None, min_recall: float | None = None,
                review_capacity: int | None = None) -> None:
    raw = load_motor_claims()
    raw_duplicate_rows = int(raw.duplicated().sum())
    if sample_size is not None and len(raw) > sample_size:
        raw = raw.sample(n=sample_size, random_state=seed)
    train, val, test = split_dataframe(raw, MOTOR_TARGET, random_state=seed)
    y_train, y_val, y_test = (part[MOTOR_TARGET].astype(int) for part in (train, val, test))
    representations = {}
    for representation, builder in (("structured_only", lambda df: build_motor_structured_features(df)),
                                    ("structured_plus_split_graph", lambda df: _motor_features_for_split(df, seed))):
        parts = {name: builder(part) for name, part in (("train", train), ("val", val), ("test", test))}
        candidate, experiment = _fit_validation_candidate(
            lambda params: MotorFraudModel(params=params), [{}, {"max_depth": 3}],
            parts["train"], y_train, parts["val"], y_val,
        )
        representations[representation] = (experiment["selected_validation_average_precision"], candidate, experiment, parts)
    chosen_name, (chosen_ap, model, experiment, parts) = max(
        representations.items(), key=lambda item: item[1][0]
    )
    experiment["representation_comparison"] = {
        name: {"validation_average_precision": result[0]} for name, result in representations.items()
    }
    experiment["selected_representation"] = chosen_name
    out = save_motor_model(model, output_dir)
    _write_json(out, "validation_experiment.json", experiment)
    _write_json(out, "data_audit.json", {
        "dataset_rows": int(len(raw)), "exact_duplicate_rows": raw_duplicate_rows,
        "native_claim_identifier": None,
        "split_strategy": "stratified row split; source has no claim identifier",
        "selected_features": "structured only" if chosen_name == "structured_only" else "structured plus split-local synthetic graph",
        "synthetic_graph_representation_comparison": experiment["representation_comparison"],
    })
    _evaluate_and_save(model, parts["val"], y_val, parts["test"], y_test, out, objective,
                       min_precision, min_recall, review_capacity)
    print(f"motor row prevalence: train={y_train.mean():.3%}, validation={y_val.mean():.3%}, test={y_test.mean():.3%}")
    print(f"motor selected representation: {chosen_name} (validation AP={chosen_ap:.4f}); graph features are split-local")


def train_health(output_dir: Path, inpatient_sample_size: int | None, outpatient_sample_size: int | None,
                 seed: int, objective: str = "max_f1", min_precision: float | None = None,
                 min_recall: float | None = None, review_capacity: int | None = None) -> None:
    tables = load_health_train_tables()
    source_claim_ids = [tables.inpatient["ClaimID"].astype(str), tables.outpatient["ClaimID"].astype(str)]
    source_id_overlap = len(set(source_claim_ids[0]) & set(source_claim_ids[1]))
    duplicate_ip_ids = int(tables.inpatient["ClaimID"].duplicated().sum())
    duplicate_op_ids = int(tables.outpatient["ClaimID"].duplicated().sum())
    inpatient = tables.inpatient
    outpatient = tables.outpatient
    if inpatient_sample_size is not None and len(inpatient) > inpatient_sample_size:
        inpatient = inpatient.sample(n=inpatient_sample_size, random_state=seed)
    if outpatient_sample_size is not None and len(outpatient) > outpatient_sample_size:
        outpatient = outpatient.sample(n=outpatient_sample_size, random_state=seed)
    claim_labels = build_health_target(inpatient, outpatient, tables.main)
    claims = pd.concat([inpatient, outpatient], ignore_index=True, sort=False)
    providers = tables.main[tables.main[HEALTH_PROVIDER_COL].isin(claims[HEALTH_PROVIDER_COL])]
    split_map = split_health_providers(providers, HEALTH_PROVIDER_COL, HEALTH_PROVIDER_LABEL_COL, random_state=seed)
    split = assign_claims_to_provider_splits(claims, HEALTH_PROVIDER_COL, split_map).to_numpy()
    split_integrity = pd.DataFrame({HEALTH_PROVIDER_COL: claims[HEALTH_PROVIDER_COL].to_numpy(), "split": split})
    provider_split_overlap = int((split_integrity.groupby(HEALTH_PROVIDER_COL)["split"].nunique() > 1).sum())
    if provider_split_overlap:
        raise ValueError(f"provider split integrity failed: {provider_split_overlap} providers cross splits")
    structured = build_health_structured_features(inpatient, outpatient, tables.beneficiary)
    labeled = structured.merge(claim_labels, on="claim_key", validate="one_to_one")
    # claim labels preserve combined inpatient/outpatient order; keys provide the alignment.
    split_by_key = pd.DataFrame({"claim_key": [f"claim:{c}" for c in claims["ClaimID"]], "split": split})
    labeled = labeled.merge(split_by_key, on="claim_key", validate="one_to_one")
    partitions = {name: labeled[labeled["split"] == name].drop(columns=["target", "split"])
                  for name in ("train", "val", "test")}
    labels = {name: labeled.loc[labeled["split"] == name, "target"].astype(int) for name in partitions}
    model, experiment = _fit_validation_candidate(
        lambda params: HealthFraudModel(**params), [{}, {"max_depth": 3}],
        partitions["train"], labels["train"], partitions["val"], labels["val"],
    )
    out = save_health_model(model, output_dir)
    _write_json(out, "validation_experiment.json", experiment)
    provider_prevalence = tables.main[HEALTH_PROVIDER_LABEL_COL].eq("Yes").mean()
    sampled_provider_prevalence = providers[HEALTH_PROVIDER_LABEL_COL].eq("Yes").mean()
    val_prevalence = labels["val"].mean()
    _write_json(out, "data_audit.json", {
        "labeled_provider_count": int(tables.main[HEALTH_PROVIDER_COL].nunique()),
        "full_provider_positive_prevalence": float(provider_prevalence),
        "sampled_claim_count": int(len(labeled)),
        "sampled_claim_proxy_positive_prevalence": float(labeled["target"].mean()),
        "validation_claim_proxy_positive_prevalence": float(val_prevalence),
        "inpatient_duplicate_claim_ids": duplicate_ip_ids,
        "outpatient_duplicate_claim_ids": duplicate_op_ids,
        "claim_id_overlap_between_sources": source_id_overlap,
        "providers_crossing_splits": provider_split_overlap,
        "label_semantics": "PotentialFraud is provider-derived and inherited by claims; not verified claim-level fraud",
    })
    _evaluate_and_save(model, partitions["val"], labels["val"], partitions["test"], labels["test"], out,
                       objective, min_precision, min_recall, review_capacity)
    print(
        f"health PotentialFraud provider prevalence: full labeled provider table={provider_prevalence:.3%}; "
        f"providers represented in sampled claims={sampled_provider_prevalence:.3%}; "
        f"sampled claim-level proxy={labeled['target'].mean():.3%}; "
        f"validation claim-level proxy={val_prevalence:.3%}"
    )
    print("The claim-level rate is higher because claims inherit provider labels and the sample weights providers by claim volume.")
    print("Health target is a provider-derived PotentialFraud proxy, not verified claim-level fraud.")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["health", "motor", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motor-sample-size", type=int, default=20000)
    parser.add_argument("--health-inpatient-sample-size", type=int, default=8000)
    parser.add_argument("--health-outpatient-sample-size", type=int, default=20000)
    parser.add_argument("--threshold-objective", choices=["max_f1", "recall_at_precision", "precision_at_recall", "review_capacity"], default="max_f1")
    parser.add_argument("--min-precision", type=float)
    parser.add_argument("--min-recall", type=float)
    parser.add_argument("--review-capacity", type=int)
    args = parser.parse_args(argv)
    try:
        opts = (args.threshold_objective, args.min_precision, args.min_recall, args.review_capacity)
        if args.domain in ("motor", "both"):
            train_motor(args.output_dir / "motor", args.motor_sample_size, args.seed, *opts)
        if args.domain in ("health", "both"):
            train_health(args.output_dir / "health", args.health_inpatient_sample_size,
                         args.health_outpatient_sample_size, args.seed, *opts)
    except (DatasetFileMissingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
