"""Fit HealthFraudModel/MotorFraudModel on real data and save them for the Streamlit app.

The app (`app/`) only loads already-trained model artifacts (see
`docs/model_persistence.md`, `docs/app_ui.md`) -- it never fits in-process.
This script is the one-time (or re-run-when-you-want-a-fresh-model) step
that produces those artifacts.

Both domains train on a bounded, seeded sample of the real raw data, not
the full dataset -- kept bounded for training-run turnaround, not because
graph centrality can't scale anymore (`src/graph/features.py`'s centrality
now auto-switches to an approximate/cached computation above 1,500 graph
nodes; see `docs/graph.md`). Increase `--motor-sample-size` /
`--health-inpatient-sample-size` / `--health-outpatient-sample-size` for a
better-fit (slower) model; `data/raw/` must be present locally either way
(see `docs/data_contract.md`).

Usage:
    ./.venv/Scripts/python.exe scripts/train_models.py --domain both
    ./.venv/Scripts/python.exe scripts/train_models.py --domain motor --motor-sample-size 5000
    ./.venv/Scripts/python.exe scripts/train_models.py --domain health \\
        --health-inpatient-sample-size 400 --health-outpatient-sample-size 900

Writes, per domain, into `<output-dir>/<domain>/`:
    model.json           -- via src.models.persistence
    metadata.json         -- via src.models.persistence
    eval_metrics.json     -- Accuracy/Precision/Recall/F1/ROC-AUC on a held-out
                              validation split (src.models.evaluation), read by
                              app/model_loader.py's load_eval_metrics()
    confusion_matrix.json -- TN/FP/FN/TP on the same held-out validation split
                              (src.models.evaluation.evaluate_confusion_matrix);
                              not read by the app today, informational only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

if __name__ == "__main__" and __package__ is None:
    # Allow `python scripts/train_models.py` directly (not just `-m`) by
    # putting the project root, not this script's own directory, on
    # sys.path so the `src`/`app` packages resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data import preprocessing
from src.data.loaders import DatasetFileMissingError, load_health_train_tables, load_motor_claims
from src.data.schemas import (
    HEALTH_PROVIDER_COL,
    HEALTH_PROVIDER_LABEL_COL,
    MOTOR_TARGET,
    PROJECT_ROOT,
)
from src.data.splits import assign_claims_to_provider_splits, split_dataframe, split_health_providers
from src.data.synthetic import build_motor_synthetic_augmentation
from src.features.fusion import fuse_health_claims, fuse_motor_claims
from src.features.structured import CLAIM_KEY_COL
from src.models.evaluation import (
    ClassificationMetrics,
    ConfusionMatrix,
    evaluate_confusion_matrix,
    evaluate_model,
)
from src.models.health import HealthFraudModel
from src.models.motor import MotorFraudModel
from src.models.persistence import save_health_model, save_motor_model

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"
DEFAULT_SEED = 42

DEFAULT_MOTOR_SAMPLE_SIZE = 3000
DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE = 300
DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE = 700

EVAL_METRICS_FILENAME = "eval_metrics.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.json"


def _sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _write_eval_metrics(directory: Path, metrics: Optional[ClassificationMetrics]) -> None:
    if metrics is None:
        return
    (directory / EVAL_METRICS_FILENAME).write_text(
        json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
    )


def _write_confusion_matrix(directory: Path, confusion: Optional[ConfusionMatrix]) -> None:
    if confusion is None:
        return
    (directory / CONFUSION_MATRIX_FILENAME).write_text(
        json.dumps(confusion.to_dict(), indent=2), encoding="utf-8"
    )


def train_motor(output_dir: Path, sample_size: int, seed: int) -> None:
    print(f"[motor] loading data/raw/motor/fraud_oracle.csv ...")
    raw = load_motor_claims()
    sampled = _sample(raw, sample_size, seed)
    print(f"[motor] training on a sample of {len(sampled)}/{len(raw)} claims")

    augmentation = build_motor_synthetic_augmentation(motor_df=sampled, seed=seed)
    claims = augmentation.claims

    print("[motor] building entity graph + fused feature matrix (this is the slow step)...")
    from src.graph.construction import build_motor_graph

    graph = build_motor_graph(claims)
    X, y = fuse_motor_claims(claims, graph)

    train_df, val_df, _test_df = split_dataframe(claims, stratify_col=MOTOR_TARGET, random_state=seed)
    train_keys = {f"claim:{idx}" for idx in train_df.index}
    val_keys = {f"claim:{idx}" for idx in val_df.index}
    train_mask = X[CLAIM_KEY_COL].isin(train_keys)
    val_mask = X[CLAIM_KEY_COL].isin(val_keys)

    print(f"[motor] fitting on {int(train_mask.sum())} training claims...")
    model = MotorFraudModel()
    model.fit(X[train_mask], y[train_mask])

    metrics = evaluate_model(model, X[val_mask], y[val_mask]) if val_mask.any() else None
    confusion = evaluate_confusion_matrix(model, X[val_mask], y[val_mask]) if val_mask.any() else None
    if metrics is not None:
        print(f"[motor] validation metrics ({int(val_mask.sum())} claims): {metrics.to_dict()}")
        print(f"[motor] confusion matrix: {confusion.to_dict()}")
    else:
        print("[motor] no validation claims available -- skipping evaluation")

    out_dir = save_motor_model(model, output_dir)
    _write_eval_metrics(out_dir, metrics)
    _write_confusion_matrix(out_dir, confusion)
    print(f"[motor] saved model to {out_dir}")


def train_health(
    output_dir: Path, inpatient_sample_size: int, outpatient_sample_size: int, seed: int
) -> None:
    print("[health] loading data/raw/health/Train*.csv ...")
    tables = load_health_train_tables()
    inpatient = _sample(tables.inpatient, inpatient_sample_size, seed)
    outpatient = _sample(tables.outpatient, outpatient_sample_size, seed)
    print(
        f"[health] training on a sample of {len(inpatient)}/{len(tables.inpatient)} inpatient + "
        f"{len(outpatient)}/{len(tables.outpatient)} outpatient claims"
    )

    claims = preprocessing.build_health_claims_table(inpatient, outpatient)

    print("[health] building entity graph + fused feature matrix (this is the slow step)...")
    from src.graph.construction import build_health_graph

    graph = build_health_graph(claims)
    X, y = fuse_health_claims(inpatient, outpatient, tables.beneficiary, tables.main, graph)

    # Provider-grouped split, restricted to the providers actually present in
    # this sample (see docs/data_contract.md section 10 -- PotentialFraud is
    # provider-level, so claims must never be split independently of their
    # provider).
    providers_in_sample = tables.main[tables.main[HEALTH_PROVIDER_COL].isin(claims[HEALTH_PROVIDER_COL])]
    provider_split_map = split_health_providers(
        providers_in_sample, HEALTH_PROVIDER_COL, HEALTH_PROVIDER_LABEL_COL, random_state=seed
    )
    claim_split = assign_claims_to_provider_splits(claims, HEALTH_PROVIDER_COL, provider_split_map)
    train_keys = {f"claim:{cid}" for cid in claims.loc[claim_split == "train", "ClaimID"]}
    val_keys = {f"claim:{cid}" for cid in claims.loc[claim_split == "val", "ClaimID"]}
    train_mask = X[CLAIM_KEY_COL].isin(train_keys)
    val_mask = X[CLAIM_KEY_COL].isin(val_keys)

    print(f"[health] fitting on {int(train_mask.sum())} training claims...")
    model = HealthFraudModel()
    model.fit(X[train_mask], y[train_mask])

    metrics = evaluate_model(model, X[val_mask], y[val_mask]) if val_mask.any() else None
    confusion = evaluate_confusion_matrix(model, X[val_mask], y[val_mask]) if val_mask.any() else None
    if metrics is not None:
        print(f"[health] validation metrics ({int(val_mask.sum())} claims): {metrics.to_dict()}")
        print(f"[health] confusion matrix: {confusion.to_dict()}")
    else:
        print("[health] no validation claims available -- skipping evaluation")

    out_dir = save_health_model(model, output_dir)
    _write_eval_metrics(out_dir, metrics)
    _write_confusion_matrix(out_dir, confusion)
    print(f"[health] saved model to {out_dir}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=["health", "motor", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--motor-sample-size", type=int, default=DEFAULT_MOTOR_SAMPLE_SIZE)
    parser.add_argument(
        "--health-inpatient-sample-size", type=int, default=DEFAULT_HEALTH_INPATIENT_SAMPLE_SIZE
    )
    parser.add_argument(
        "--health-outpatient-sample-size", type=int, default=DEFAULT_HEALTH_OUTPATIENT_SAMPLE_SIZE
    )
    args = parser.parse_args(argv)

    try:
        if args.domain in ("motor", "both"):
            train_motor(args.output_dir / "motor", args.motor_sample_size, args.seed)
        if args.domain in ("health", "both"):
            train_health(
                args.output_dir / "health",
                args.health_inpatient_sample_size,
                args.health_outpatient_sample_size,
                args.seed,
            )
    except DatasetFileMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
